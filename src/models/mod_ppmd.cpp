// ppmd is written by Dmitry Shkarin.
// mod_ppmd is adapted from ppmd by Eugene Shelwien.
// This file is adapted from mod_ppmd_v2: http://encode.su/threads/2515-mod_ppmd

#include "mod_ppmd.h"

#include <cstring>
#include <numeric>

#include "../memory-interface.h"

namespace PPMD {

template <class T>
T Min(T x, T y) {
  return (x < y) ? x : y;
}
template <class T>
T Max(T x, T y) {
  return (x > y) ? x : y;
}

#pragma pack(1)

typedef unsigned short word;
typedef unsigned int uint;
typedef unsigned char byte;
typedef unsigned long long qword;

const int ORealMAX = 256;

static signed char EscCoef[12] = {16, -10, 1,  51, 14,  89,
                                  23, 35,  64, 26, -42, 43};

// Tabulated escapes for exponential symbol distribution
static const byte ExpEscape[16] = {51, 43, 18, 12, 11, 9, 8, 7,
                                   6,  5,  4,  3,  3,  2, 2, 2};

class ppmd_Model : public MemoryInterface {
 public:
  typedef unsigned short word;
  typedef unsigned int uint;
  typedef unsigned char byte;
  typedef unsigned long long qword;
  bool enable_learn = true;

  enum { SCALE = 1 << 15 };

  enum {
    UNIT_SIZE = 12,
    N1 = 4,
    N2 = 4,
    N3 = 4,
    N4 = (128 + 3 - 1 * N1 - 2 * N2 - 3 * N3) / 4,
    N_INDEXES = N1 + N2 + N3 + N4
  };

  byte* HeapStart;
  typedef byte* pbyte;
  uint Ptr2Indx(void* p) { return pbyte(p) - HeapStart; }
  void* Indx2Ptr(uint indx) { return indx + HeapStart; }

  struct _MEM_BLK {
    uint Stamp;
    uint NextIndx;
    uint NU;
  };

  struct BLK_NODE {
    uint Stamp;
    uint NextIndx;
    int avail() const { return (NextIndx != 0); }
  };

  BLK_NODE* getNext(BLK_NODE* This) {
    return (BLK_NODE*)Indx2Ptr(This->NextIndx);
  }

  void setNext(BLK_NODE* This, BLK_NODE* p) { This->NextIndx = Ptr2Indx(p); }

  void link(BLK_NODE* This, BLK_NODE* p) {
    p->NextIndx = This->NextIndx;
    setNext(This, p);
  }

  void unlink(BLK_NODE* This) { This->NextIndx = getNext(This)->NextIndx; }

  void* remove(BLK_NODE* This) {
    BLK_NODE* p = getNext(This);
    unlink(This);
    This->Stamp--;
    return p;
  }

  void insert(BLK_NODE* This, void* pv, int NU) {
    BLK_NODE* p = (BLK_NODE*)pv;
    link(This, p);
    p->Stamp = ~uint(0);
    ((_MEM_BLK&)*p).NU = NU;
    This->Stamp++;
  }

  struct MEM_BLK : public BLK_NODE {
    uint NU;
  };

  typedef BLK_NODE* pBLK_NODE;

  typedef MEM_BLK* pMEM_BLK;

  BLK_NODE BList[N_INDEXES + 1];

  uint GlueCount;
  uint GlueCount1;
  qword SubAllocatorSize;
  byte* pText;
  byte* UnitsStart;
  byte* LoUnit;
  byte* HiUnit;
  byte* AuxUnit;

  uint U2B(uint NU) { return 8 * NU + 4 * NU; }

  int StartSubAllocator(qword SASize) {
    qword t = SASize << 20U;
    HeapStart = new byte[t];
    if (HeapStart == NULL) return 0;
    SubAllocatorSize = t;
    return 1;
  }

  void InitSubAllocator() {
    memset(BList, 0, sizeof(BList));
    HiUnit = (pText = HeapStart) + SubAllocatorSize;
    qword Diff = SubAllocatorSize / 8 / UNIT_SIZE * 7 * UNIT_SIZE;
    LoUnit = UnitsStart = HiUnit - Diff;
    GlueCount = GlueCount1 = 0;
  }

  qword GetUsedMemory() {
    int i;
    qword RetVal = SubAllocatorSize - (HiUnit - LoUnit) - (UnitsStart - pText);
    for (i = 0; i < N_INDEXES; i++) {
      RetVal -= qword(Indx2Units[i] * BList[i].Stamp) * 12;
    }
    return RetVal;
  }

  void StopSubAllocator() {
    if (SubAllocatorSize) {
      SubAllocatorSize = 0;
      delete[] HeapStart;
    }
  }

  void GlueFreeBlocks() {
    uint i, k, sz;
    MEM_BLK s0;
    pMEM_BLK p, p0 = &s0, p1;

    if (LoUnit != HiUnit) LoUnit[0] = 0;

    for (p0->NextIndx = 0, i = 0; i <= N_INDEXES; i++) {
      while (BList[i].avail()) {
        p = (MEM_BLK*)remove(&BList[i]);
        if (p->NU) {
          while (p1 = p + p->NU, p1->Stamp == ~uint(0)) {
            p->NU += p1->NU;
            p1->NU = 0;
          }
          link(p0, p);
          p0 = p;
        }
      }
    }

    while (s0.avail()) {
      p = (MEM_BLK*)remove(&s0);
      sz = p->NU;
      if (sz) {
        for (; sz > 128; sz -= 128, p += 128)
          insert(&BList[N_INDEXES - 1], p, 128);
        i = Units2Indx[sz - 1];
        if (Indx2Units[i] != sz) {
          k = sz - Indx2Units[--i];
          insert(&BList[k - 1], p + (sz - k), k);
        }
        insert(&BList[i], p, Indx2Units[i]);
      }
    }

    GlueCount = 1 << (13 + GlueCount1++);
  }

  void SplitBlock(void* pv, uint OldIndx, uint NewIndx) {
    uint i, k, UDiff = Indx2Units[OldIndx] - Indx2Units[NewIndx];
    byte* p = ((byte*)pv) + U2B(Indx2Units[NewIndx]);
    i = Units2Indx[UDiff - 1];
    if (Indx2Units[i] != UDiff) {
      k = Indx2Units[--i];
      insert(&BList[i], p, k);
      p += U2B(k);
      UDiff -= k;
    }
    insert(&BList[Units2Indx[UDiff - 1]], p, UDiff);
  }

  void* AllocUnitsRare(uint indx) {
    uint i = indx;
    do {
      if (++i == N_INDEXES) {
        if (!GlueCount--) {
          GlueFreeBlocks();
          if (BList[i = indx].avail()) return remove(&BList[i]);
        } else {
          i = U2B(Indx2Units[indx]);
          return (UnitsStart - pText > i) ? UnitsStart -= i : NULL;
        }
      }
    } while (!BList[i].avail());

    void* RetVal = remove(&BList[i]);
    SplitBlock(RetVal, i, indx);

    return RetVal;
  }

  void* AllocUnits(uint NU) {
    uint indx = Units2Indx[NU - 1];
    if (BList[indx].avail()) return remove(&BList[indx]);
    void* RetVal = LoUnit;
    LoUnit += U2B(Indx2Units[indx]);
    if (LoUnit <= HiUnit) return RetVal;
    LoUnit -= U2B(Indx2Units[indx]);
    return AllocUnitsRare(indx);
  }

  void* AllocContext() {
    if (HiUnit != LoUnit) return HiUnit -= UNIT_SIZE;
    return BList->avail() ? remove(BList) : AllocUnitsRare(0);
  }

  void FreeUnits(void* ptr, uint NU) {
    uint indx = Units2Indx[NU - 1];
    insert(&BList[indx], ptr, Indx2Units[indx]);
  }

  void FreeUnit(void* ptr) {
    int i = (byte*)ptr > UnitsStart + 128 * 1024 ? 0 : N_INDEXES;
    insert(&BList[i], ptr, 1);
  }

  void UnitsCpy(void* Dest, void* Src, uint NU) { memcpy(Dest, Src, 12 * NU); }

  void* ExpandUnits(void* OldPtr, uint OldNU) {
    uint i0 = Units2Indx[OldNU - 1];
    uint i1 = Units2Indx[OldNU - 1 + 1];
    if (i0 == i1) return OldPtr;
    void* ptr = AllocUnits(OldNU + 1);
    if (ptr) {
      UnitsCpy(ptr, OldPtr, OldNU);
      insert(&BList[i0], OldPtr, OldNU);
    }
    return ptr;
  }

  void* ShrinkUnits(void* OldPtr, uint OldNU, uint NewNU) {
    uint i0 = Units2Indx[OldNU - 1];
    uint i1 = Units2Indx[NewNU - 1];
    if (i0 == i1) return OldPtr;
    if (BList[i1].avail()) {
      void* ptr = remove(&BList[i1]);
      UnitsCpy(ptr, OldPtr, NewNU);
      insert(&BList[i0], OldPtr, Indx2Units[i0]);
      return ptr;
    } else {
      SplitBlock(OldPtr, i0, i1);
      return OldPtr;
    }
  }

  void* MoveUnitsUp(void* OldPtr, uint NU) {
    uint indx = Units2Indx[NU - 1];
    PrefetchData(OldPtr);
    if ((byte*)OldPtr > UnitsStart + 128 * 1024 ||
        (BLK_NODE*)OldPtr > getNext(&BList[indx]))
      return OldPtr;

    void* ptr = remove(&BList[indx]);
    UnitsCpy(ptr, OldPtr, NU);

    insert(&BList[N_INDEXES], OldPtr, Indx2Units[indx]);

    return ptr;
  }

  void PrepareTextArea() {
    AuxUnit = (byte*)AllocContext();
    if (!AuxUnit) {
      AuxUnit = UnitsStart;
    } else {
      if (AuxUnit == UnitsStart) AuxUnit = (UnitsStart += UNIT_SIZE);
    }
  }

  void ExpandTextArea() {
    BLK_NODE* p;
    uint Count[N_INDEXES], i = 0;
    memset(Count, 0, sizeof(Count));

    if (AuxUnit != UnitsStart) {
      if (*(uint*)AuxUnit != ~uint(0))
        UnitsStart += UNIT_SIZE;
      else
        insert(BList, AuxUnit, 1);
    }

    while ((p = (BLK_NODE*)UnitsStart)->Stamp == ~uint(0)) {
      MEM_BLK* pm = (MEM_BLK*)p;
      UnitsStart = (byte*)(pm + pm->NU);
      Count[Units2Indx[pm->NU - 1]]++;
      i++;
      pm->Stamp = 0;
    }

    if (i) {
      for (p = BList + N_INDEXES; p->NextIndx; p = getNext(p)) {
        while (p->NextIndx && !getNext(p)->Stamp) {
          Count[Units2Indx[((MEM_BLK*)getNext(p))->NU - 1]]--;
          unlink(p);
          BList[N_INDEXES].Stamp--;
        }
        if (!p->NextIndx) break;
      }

      for (i = 0; i < N_INDEXES; i++) {
        for (p = BList + i; Count[i] != 0; p = getNext(p)) {
          while (!getNext(p)->Stamp) {
            unlink(p);
            BList[i].Stamp--;
            if (!--Count[i]) break;
          }
        }
      }
    }
  }

  static const int MAX_O = ORealMAX;  // maximum allowed model order

  template <class T>
  T CLAMP(const T& X, const T& LoX, const T& HiX) {
    return (X >= LoX) ? ((X <= HiX) ? (X) : (HiX)) : (LoX);
  }

  template <class T>
  void SWAP(T& t1, T& t2) {
    T tmp = t1;
    t1 = t2;
    t2 = tmp;
  }

  void PrefetchData(void* Addr) { *(volatile byte*)Addr; }

  enum { UP_FREQ = 5 };

  byte Indx2Units[N_INDEXES];
  byte Units2Indx[128];  // constants

  byte NS2BSIndx[256];
  byte QTable[260];

  // constants initialization
  void PPMD_STARTUP(void) {
    int i, k, m, Step;

    for (i = 0, k = 1; i < N1; i++, k += 1) Indx2Units[i] = k;
    for (k++; i < N1 + N2; i++, k += 2) Indx2Units[i] = k;
    for (k++; i < N1 + N2 + N3; i++, k += 3) Indx2Units[i] = k;
    for (k++; i < N1 + N2 + N3 + N4; i++, k += 4) Indx2Units[i] = k;

    for (k = 0, i = 0; k < 128; k++) {
      i += Indx2Units[i] < k + 1;
      Units2Indx[k] = i;
    }

    NS2BSIndx[0] = 2 * 0;  //-V525
    NS2BSIndx[1] = 2 * 1;
    NS2BSIndx[2] = 2 * 1;
    memset(NS2BSIndx + 3, 2 * 2, 26);
    memset(NS2BSIndx + 29, 2 * 3, 256 - 29);

    for (i = 0; i < UP_FREQ; i++) QTable[i] = i;

    for (m = i = UP_FREQ, k = Step = 1; i < 260; i++) {
      QTable[i] = m;
      if (!--k) k = ++Step, m++;
    }
  }

  enum { MAX_FREQ = 124, O_BOUND = 9 };

  struct PPM_CONTEXT;

  struct STATE {
    byte Symbol;
    byte Freq;
    uint iSuccessor;
  };

  PPM_CONTEXT* getSucc(STATE* This) {
    return (PPM_CONTEXT*)Indx2Ptr(This->iSuccessor);
  }

  void SWAP(STATE& s1, STATE& s2) {
    word t1 = (word&)s1;
    uint t2 = s1.iSuccessor;
    (word&)s1 = (word&)s2;
    s1.iSuccessor = s2.iSuccessor;
    (word&)s2 = t1;
    s2.iSuccessor = t2;
  }

  struct PPM_CONTEXT {
    byte NumStats;
    byte Flags;
    word SummFreq;
    uint iStats;
    uint iSuffix;

    STATE& oneState() const { return (STATE&)SummFreq; }
  };

  STATE* getStats(PPM_CONTEXT* This) { return (STATE*)Indx2Ptr(This->iStats); }

  PPM_CONTEXT* suff(PPM_CONTEXT* This) {
    return (PPM_CONTEXT*)Indx2Ptr(This->iSuffix);
  }

  int _MaxOrder, _CutOff, _MMAX;
  uint _filesize;
  int OrderFall;

  STATE* FoundState;  // found next state transition
  PPM_CONTEXT* MaxContext;

  uint EscCount;
  uint CharMask[256];

  int BSumm;
  int RunLength;
  int InitRL;

  enum {
    INT_BITS = 7,
    PERIOD_BITS = 7,
    TOT_BITS = INT_BITS + PERIOD_BITS,
    INTERVAL = 1 << INT_BITS,
    BIN_SCALE = 1 << TOT_BITS,
    ROUND = 16
  };

  // SEE-contexts for PPM-contexts with masked symbols
  struct SEE2_CONTEXT {
    word Summ;
    byte Shift;
    byte Count;

    void init(uint InitVal) {
      Shift = PERIOD_BITS - 4;
      Summ = InitVal << Shift;
      Count = 7;
    }

    uint getMean() { return Summ >> Shift; }

    void update() {
      if (--Count == 0) setShift_rare();
    }

    void setShift_rare() {
      uint i = Summ >> Shift;
      i = PERIOD_BITS - (i > 40) - (i > 280) - (i > 1020);
      if (i < Shift) {
        Summ >>= 1;
        Shift--;
      } else if (i > Shift) {
        Summ <<= 1;
        Shift++;
      }
      Count = 5 << Shift;
    }
  };

  int NumMasked;

  STATE* rescale(PPM_CONTEXT& q, int OrderFall, STATE* FoundState) {
    STATE tmp;
    STATE* p;
    STATE* p1;

    q.Flags &= 0x14;

    // move the current node to rank0
    p1 = getStats(&q);
    tmp = FoundState[0];
    for (p = FoundState; p != p1; p--) p[0] = p[-1];
    p1[0] = tmp;

    int of = (OrderFall != 0);
    int a, i;
    int f0 = p->Freq;
    int sf = q.SummFreq;
    int EscFreq = sf - f0;
    q.SummFreq = p->Freq = (f0 + of) >> 1;

    // sort symbols by freqs
    for (i = 0; i < q.NumStats; i++) {
      p++;
      a = p->Freq;
      EscFreq -= a;
      a = (a + of) >> 1;
      p->Freq = a;
      q.SummFreq += a;
      if (a) q.Flags |= 0x08 * (p->Symbol >= 0x40);
      if (a > p[-1].Freq) {
        tmp = p[0];
        for (p1 = p; tmp.Freq > p1[-1].Freq; p1--) p1[0] = p1[-1];
        p1[0] = tmp;
      }
    }

    // remove the zero freq nodes
    if (p->Freq == 0) {
      for (i = 0; p->Freq == 0; i++, p--);
      EscFreq += i;
      a = (q.NumStats + 2) >> 1;
      if ((q.NumStats -= i) == 0) {
        tmp = getStats(&q)[0];
        tmp.Freq = Min(MAX_FREQ / 3, (2 * tmp.Freq + EscFreq - 1) / EscFreq);
        q.Flags &= 0x18;
        FreeUnits(getStats(&q), a);
        q.oneState() = tmp;
        FoundState = &q.oneState();
        return FoundState;
      }
      q.iStats = Ptr2Indx(ShrinkUnits(getStats(&q), a, (q.NumStats + 2) >> 1));
    }

    // some weird magic
    q.SummFreq += (EscFreq + 1) >> 1;
    if (OrderFall || (q.Flags & 0x04) == 0) {
      a = (sf -= EscFreq) - f0;
      a = CLAMP(uint((f0 * q.SummFreq - sf * getStats(&q)->Freq + a - 1) / a),
                2U, MAX_FREQ / 2U - 18U);
    } else {
      a = 2;
    }

    (FoundState = getStats(&q))->Freq += a;
    q.SummFreq += a;
    q.Flags |= 0x04;

    return FoundState;
  }

  void AuxCutOff(STATE* p, int Order, int MaxOrder) {
    if (Order < MaxOrder) {
      PrefetchData(getSucc(p));
      p->iSuccessor = cutOff(getSucc(p)[0], Order + 1, MaxOrder);
    } else {
      p->iSuccessor = 0;
    }
  }

  uint cutOff(PPM_CONTEXT& q, int Order, int MaxOrder) {
    int i, tmp, EscFreq, Scale;
    STATE* p;
    STATE* p0;

    // for binary context, just cut off the successors
    if (q.NumStats == 0) {
      int flag = 1;
      p = &q.oneState();
      if ((byte*)getSucc(p) >= UnitsStart) {
        AuxCutOff(p, Order, MaxOrder);
        if (p->iSuccessor || Order < O_BOUND) flag = 0;
      }
      if (flag) {
        FreeUnit(&q);
        return 0;
      }

    } else {
      tmp = (q.NumStats + 2) >> 1;
      p0 = (STATE*)MoveUnitsUp(getStats(&q), tmp);
      q.iStats = Ptr2Indx(p0);

      // cut the branches with links to text
      for (i = q.NumStats, p = &p0[i]; p >= p0; p--) {
        if ((byte*)getSucc(p) < UnitsStart) {
          p[0].iSuccessor = 0;
          SWAP(p[0], p0[i--]);
        } else
          AuxCutOff(p, Order, MaxOrder);
      }

      // if something was cut
      if (i != q.NumStats && Order > 0) {
        q.NumStats = i;
        p = p0;
        if (i < 0) {
          FreeUnits(p, tmp);
          FreeUnit(&q);
          return 0;
        }
        if (i == 0) {
          q.Flags = (q.Flags & 0x10) + 0x08 * (p[0].Symbol >= 0x40);
          p[0].Freq = 1 + (2 * (p[0].Freq - 1)) / (q.SummFreq - p[0].Freq);
          q.oneState() = p[0];
          FreeUnits(p, tmp);
        } else {
          p = (STATE*)ShrinkUnits(p0, tmp, (i + 2) >> 1);
          q.iStats = Ptr2Indx(p);
          Scale = (q.SummFreq > 16 * i);
          q.Flags = (q.Flags & (0x10 + 0x04 * Scale));
          if (Scale) {
            EscFreq = q.SummFreq;
            q.SummFreq = 0;
            for (i = 0; i <= q.NumStats; i++) {
              EscFreq -= p[i].Freq;
              p[i].Freq = (p[i].Freq + 1) >> 1;
              q.SummFreq += p[i].Freq;
              q.Flags |= 0x08 * (p[i].Symbol >= 0x40);
            };
            EscFreq = (EscFreq + 1) >> 1;
            q.SummFreq += EscFreq;
          } else {
            for (i = 0; i <= q.NumStats; i++)
              q.Flags |= 0x08 * (p[i].Symbol >= 0x40);
          }
        }
      }
    }

    if ((byte*)&q == UnitsStart) {
      // if this is a root, copy it
      UnitsCpy(AuxUnit, &q, 1);
      return Ptr2Indx(AuxUnit);
    } else {
      // if suffix is root, switch the pointer
      if ((byte*)suff(&q) == UnitsStart) q.iSuffix = Ptr2Indx(AuxUnit);
    }

    return Ptr2Indx(&q);
  }

  void StartModelRare(void) {
    int i, k, s;
    byte i2f[25];

    memset(CharMask, 0, sizeof(CharMask));
    EscCount = 1;

    // we are in solid mode
    if (_MaxOrder < 2) {
      OrderFall = _MaxOrder;
      for (PPM_CONTEXT* pc = MaxContext; pc->iSuffix != 0; pc = suff(pc))
        OrderFall--;
      return;
    }

    OrderFall = _MaxOrder;

    InitSubAllocator();

    InitRL = -((_MaxOrder < 13) ? _MaxOrder : 13);
    RunLength = InitRL;

    // alloc and init order0 context
    MaxContext = (PPM_CONTEXT*)AllocContext();
    MaxContext->NumStats = 255;
    MaxContext->SummFreq = 255 + 2;
    MaxContext->iStats = Ptr2Indx(AllocUnits(256 / 2));
    MaxContext->Flags = 0;
    MaxContext->iSuffix = 0;
    PrevSuccess = 0;

    for (i = 0; i < 256; i++) {
      getStats(MaxContext)[i].Symbol = i;
      getStats(MaxContext)[i].Freq = 1;
      getStats(MaxContext)[i].iSuccessor = 0;
    }

    // _InitSEE
    if (1) {
      // a freq for quant?
      for (k = i = 0; i < 25; i2f[i++] = k + 1)
        while (QTable[k] == i) k++;

      // bin SEE init
      for (k = 0; k < 64; k++) {
        for (s = i = 0; i < 6; i++) s += EscCoef[2 * i + ((k >> i) & 1)];
        s = 128 * CLAMP(s, 32, 256 - 32);
        for (i = 0; i < 25; i++) BinSumm[i][k] = BIN_SCALE - s / i2f[i];
      }

      // masked SEE init
      for (i = 0; i < 23; i++)
        for (k = 0; k < 32; k++) SEE2Cont[i][k].init(8 * i + 5);
    }
  }

  // model flush
  void RestoreModelRare(void) {
    STATE* p;
    pText = HeapStart;
    PPM_CONTEXT* pc = saved_pc;

    // from maxorder down, while there 2 symbols and 2nd symbol has a text
    // pointer
    for (;; MaxContext = suff(MaxContext)) {
      if ((MaxContext->NumStats == 1) && (MaxContext != pc)) {
        p = getStats(MaxContext);
        if ((byte*)(getSucc(p + 1)) >= UnitsStart) break;
      } else
        break;
      // turn a context with 2 symbols into a context with 1 symbol
      MaxContext->Flags =
          (MaxContext->Flags & 0x10) + 0x08 * (p->Symbol >= 0x40);
      p[0].Freq = (p[0].Freq + 1) >> 1;
      MaxContext->oneState() = p[0];
      MaxContext->NumStats = 0;
      FreeUnits(p, 1);
    }

    // go all the way down
    while (MaxContext->iSuffix) MaxContext = suff(MaxContext);

    AuxUnit = UnitsStart;

    ExpandTextArea();

    // free up 25% of memory
    do {
      PrepareTextArea();
      cutOff(MaxContext[0], 0,
             _MaxOrder);  // MaxContext is a tree root here, order0
      ExpandTextArea();
    } while (GetUsedMemory() > 3 * (SubAllocatorSize >> 2));

    GlueCount = GlueCount1 = 0;
    OrderFall = _MaxOrder;
  }

  PPM_CONTEXT* saved_pc;

  PPM_CONTEXT* UpdateModel(PPM_CONTEXT* MinContext) {
    byte Flag, FSymbol;
    uint ns1, ns, cf, sf, s0, FFreq;
    uint iSuccessor, iFSuccessor;
    PPM_CONTEXT* pc;
    STATE* p = NULL;

    FSymbol = FoundState->Symbol;
    FFreq = FoundState->Freq;
    iFSuccessor = FoundState->iSuccessor;

    // partial update for the suffix context
    if (MinContext->iSuffix && enable_learn) {
      pc = suff(MinContext);
      // is it binary?
      if (pc[0].NumStats) {
        p = getStats(pc);
        if (p[0].Symbol != FSymbol) {
          for (p++; p[0].Symbol != FSymbol; p++);
          if (p[0].Freq >= p[-1].Freq) SWAP(p[0], p[-1]), p--;
        }
        if (p[0].Freq < MAX_FREQ - 3) {
          cf = 2 + (FFreq < 28);
          p[0].Freq += cf;
          pc[0].SummFreq += cf;
        }
      } else {
        p = &(pc[0].oneState());
        p[0].Freq += (p[0].Freq < 14);
      }
    }
    pc = MaxContext;

    // try increasing the order
    if (!OrderFall && iFSuccessor && enable_learn) {
      FoundState->iSuccessor = CreateSuccessors(1, p, MinContext);
      if (!FoundState->iSuccessor) {
        saved_pc = pc;
        return 0;
      };
      MaxContext = getSucc(FoundState);
      return MaxContext;
    }

    if (enable_learn) *pText++ = FSymbol;
    iSuccessor = Ptr2Indx(pText);
    if (pText >= UnitsStart) {
      saved_pc = pc;
      return 0;
    };

    if (iFSuccessor) {
      if ((byte*)Indx2Ptr(iFSuccessor) < UnitsStart)
        iFSuccessor = CreateSuccessors(0, p, MinContext);
      else
        PrefetchData(Indx2Ptr(iFSuccessor));
    } else
      iFSuccessor = ReduceOrder(p, MinContext);

    if (!iFSuccessor) {
      saved_pc = pc;
      return 0;
    };

    if (!--OrderFall) {
      iSuccessor = iFSuccessor;
      pText -= (MaxContext != MinContext);
    }

    s0 = MinContext->SummFreq - FFreq;
    ns = MinContext->NumStats;
    Flag = 0x08 * (FSymbol >= 0x40);
    for (pc = MaxContext; enable_learn && pc != MinContext; pc = suff(pc)) {
      ns1 = pc[0].NumStats;
      // non-binary context?
      if (ns1) {
        // realloc table with alphabet size is odd
        if (ns1 & 1) {
          p = (STATE*)ExpandUnits(getStats(pc), (ns1 + 1) >> 1);
          if (!p) {
            saved_pc = pc;
            return 0;
          };
          pc[0].iStats = Ptr2Indx(p);
        }
        // increase escape freq (more for larger alphabet)
        pc[0].SummFreq += QTable[ns + 4] >> 3;
      } else {
        // escaped binary context
        p = (STATE*)AllocUnits(1);
        if (!p) {
          saved_pc = pc;
          return 0;
        };
        p[0] = pc[0].oneState();
        pc[0].iStats = Ptr2Indx(p);
        p[0].Freq =
            (p[0].Freq <= MAX_FREQ / 3) ? (2 * p[0].Freq - 1) : (MAX_FREQ - 15);
        // update escape
        pc[0].SummFreq =
            p[0].Freq + (ns > 1) + ExpEscape[QTable[BSumm >> 8]];  //-V602
      }

      // inheritance
      cf = (FFreq - 1) * (5 + pc[0].SummFreq);
      sf = s0 + pc[0].SummFreq;
      // this is a weighted rescaling of symbol's freq into a new context
      // (cf/sf)
      if (cf <= 3 * sf) {
        // if the new freq is too small the we increase the escape freq too
        cf = 1 + (2 * cf > sf) + (2 * cf > 3 * sf);
        pc[0].SummFreq += 4;
      } else {
        cf = 5 + (cf > 5 * sf) + (cf > 6 * sf) + (cf > 8 * sf) +
             (cf > 10 * sf) + (cf > 12 * sf);
        pc[0].SummFreq += cf;
      }

      p = getStats(pc) + (++pc[0].NumStats);
      p[0].iSuccessor = iSuccessor;
      p[0].Symbol = FSymbol;
      p[0].Freq = cf;
      pc[0].Flags |= Flag;  // flag if last added symbol was >=0x40
    }

    MaxContext = (PPM_CONTEXT*)Indx2Ptr(iFSuccessor);
    return MaxContext;
  }

  uint CreateSuccessors(uint Skip, STATE* p, PPM_CONTEXT* pc) {
    if (!enable_learn) return Ptr2Indx(pc);
    byte tmp;
    uint cf, s0;
    STATE* ps[MAX_O];
    STATE** pps = ps;

    byte sym = FoundState->Symbol;
    uint iUpBranch = FoundState->iSuccessor;

    if (!Skip) {
      *pps++ = FoundState;
      if (!pc[0].iSuffix) goto NO_LOOP;
    }

    if (p) {
      pc = suff(pc);
      goto LOOP_ENTRY;
    }

    do {
      pc = suff(pc);

      // increment current symbol's freq in lower order contexts
      // more partial updates?
      if (pc[0].NumStats) {
        // find sym node
        for (p = getStats(pc); p[0].Symbol != sym; p++);
        // increment freq if limit allows
        tmp = 2 * (p[0].Freq < MAX_FREQ - 1);
        p[0].Freq += tmp;
        pc[0].SummFreq += tmp;
      } else {
        // binary context
        p = &(pc[0].oneState());
        p[0].Freq += (!suff(pc)->NumStats & (p[0].Freq < 16));
      }

    LOOP_ENTRY:
      if (p[0].iSuccessor != iUpBranch) {
        pc = getSucc(p);
        break;
      }
      *pps++ = p;
    } while (pc[0].iSuffix);

  NO_LOOP:
    if (pps == ps) return Ptr2Indx(pc);

    // fetch a following symbol from the text buffer
    PPM_CONTEXT ct;
    ct.NumStats = 0;
    ct.Flags = 0x10 * (sym >= 0x40);
    sym = *(byte*)Indx2Ptr(iUpBranch);
    ct.oneState().iSuccessor = Ptr2Indx((byte*)Indx2Ptr(iUpBranch) + 1);
    ct.oneState().Symbol = sym;
    ct.Flags |= 0x08 * (sym >= 0x40);

    // pc is MinContext, the context used for encoding
    if (pc[0].NumStats) {
      for (p = getStats(pc); p[0].Symbol != sym; p++);
      cf = p[0].Freq - 1;
      s0 = pc[0].SummFreq - pc[0].NumStats - cf;
      cf = 1 + ((2 * cf < s0) ? (12 * cf > s0) : 2 + cf / s0);
      ct.oneState().Freq = Min<uint>(7, cf);
    } else {
      ct.oneState().Freq = pc[0].oneState().Freq;
    }

    // attach the new node to all orders
    do {
      PPM_CONTEXT* pc1 = (PPM_CONTEXT*)AllocContext();
      if (!pc1) return 0;
      ((uint*)pc1)[0] = ((uint*)&ct)[0];
      ((uint*)pc1)[1] = ((uint*)&ct)[1];
      pc1->iSuffix = Ptr2Indx(pc);
      pc = pc1;
      pps--;
      pps[0][0].iSuccessor = Ptr2Indx(pc);
    } while (pps != ps);

    return Ptr2Indx(pc);
  }

  uint ReduceOrder(STATE* p, PPM_CONTEXT* pc) {
    byte tmp;
    STATE* p1;
    PPM_CONTEXT* pc1 = pc;
    if (enable_learn) FoundState->iSuccessor = Ptr2Indx(pText);
    byte sym = FoundState->Symbol;
    uint iUpBranch = FoundState->iSuccessor;
    if (!enable_learn) iUpBranch = Ptr2Indx(pText);
    OrderFall++;

    if (p) {
      pc = suff(pc);
      goto LOOP_ENTRY;
    }

    while (1) {
      if (!pc->iSuffix) return Ptr2Indx(pc);
      pc = suff(pc);

      if (pc->NumStats) {
        for (p = getStats(pc); p[0].Symbol != sym; p++);
        tmp = 2 * (p->Freq < MAX_FREQ - 3);
        if (enable_learn) {
          p->Freq += tmp;
          pc->SummFreq += tmp;
        }
      } else {
        p = &(pc->oneState());
        if (enable_learn) {
          p->Freq += (p->Freq < 11);
        }
      }

    LOOP_ENTRY:
      if (p->iSuccessor) break;
      if (enable_learn) p->iSuccessor = iUpBranch;
      OrderFall++;
    }

    bool custom_return = false;
    if (p->iSuccessor <= iUpBranch) {
      p1 = FoundState;
      FoundState = p;
      if (enable_learn) p->iSuccessor = CreateSuccessors(0, 0, pc);
      else custom_return = true;
      FoundState = p1;
    }

    if (OrderFall == 1 && pc1 == MaxContext) {
      if (enable_learn) FoundState->iSuccessor = p->iSuccessor;
      pText--;
    }

    if (custom_return) return CreateSuccessors(0, 0, pc);
    return p->iSuccessor;
  }

  int PrevSuccess;
  word BinSumm[25][64];  // binary SEE-contexts

  template <int ProcMode>
  void processBinSymbol(PPM_CONTEXT& q, int symbol) {
    STATE& rs = q.oneState();
    int i = NS2BSIndx[suff(&q)->NumStats] + PrevSuccess + q.Flags +
            ((RunLength >> 26) & 0x20);
    word& bs = BinSumm[QTable[rs.Freq - 1]][i];
    BSumm = bs;
    bs -= (BSumm + 64) >> PERIOD_BITS;

    int flag = ProcMode ? 0 : rs.Symbol != symbol;

    if (flag) {
      CharMask[rs.Symbol] = EscCount;
      NumMasked = 0;
      PrevSuccess = 0;
      FoundState = 0;
    } else {
      bs += INTERVAL;
      if (enable_learn) rs.Freq += (rs.Freq < 196);
      RunLength++;
      PrevSuccess = 1;
      FoundState = &rs;
    }
  }

  // encode in unmasked (maxorder) context
  template <int ProcMode>
  void processSymbol1(PPM_CONTEXT& q, int symbol) {
    STATE* p = getStats(&q);

    int cnum = q.NumStats;
    int i = p[0].Symbol;
    int low = 0;
    int freq = p[0].Freq;
    int total = q.SummFreq;
    int flag;
    int count = 0;

    if (ProcMode) {
      flag = count < freq;
    } else {
      flag = i == symbol;
    }

    if (flag) {
      PrevSuccess = 0;  //(2*freq>1*total);
      if (enable_learn) {
        p[0].Freq += 4;
        q.SummFreq += 4;
      }
    } else {
      PrevSuccess = 0;

      for (low = freq, i = 1; i <= cnum; i++) {
        freq = p[i].Freq;
        flag = ProcMode ? low + freq > count : p[i].Symbol == symbol;
        if (flag) break;
        low += freq;
      }

      if (flag) {
        if (enable_learn) {
          p[i].Freq += 4;
          q.SummFreq += 4;
          if (p[i].Freq > p[i - 1].Freq) SWAP(p[i], p[i - 1]), i--;
        }
        p = &p[i];
      } else {
        if (q.iSuffix) PrefetchData(suff(&q));
        freq = total - low;
        NumMasked = cnum;
        for (i = 0; i <= cnum; i++) CharMask[p[i].Symbol] = EscCount;
        p = NULL;
      }
    }

    FoundState = p;
    if (p && (p[0].Freq > MAX_FREQ))
      FoundState = rescale(q, OrderFall, FoundState);
  }

  SEE2_CONTEXT SEE2Cont[23][32];
  SEE2_CONTEXT DummySEE2Cont;

  // encode in masked context
  template <int ProcMode>
  void processSymbol2(PPM_CONTEXT& q, int symbol) {
    byte px[256];
    STATE* p = getStats(&q);

    int c;
    int count = 0;
    int low;
    int see_freq;
    int cnum = q.NumStats;

    SEE2_CONTEXT* psee2c;
    if (cnum != 0xFF) {
      psee2c = SEE2Cont[QTable[cnum + 3] - 4];
      psee2c += (q.SummFreq > 10 * (cnum + 1));
      psee2c += 2 * (2 * cnum < suff(&q)->NumStats + NumMasked) + q.Flags;
      see_freq = psee2c->getMean() + 1;
    } else {
      psee2c = &DummySEE2Cont;
      see_freq = 1;
    }

    int flag = 0, pl;

    int i, j;
    for (i = 0, j = 0, low = 0; i <= cnum; i++) {
      c = p[i].Symbol;
      if (CharMask[c] != EscCount) {
        CharMask[c] = EscCount;
        low += p[i].Freq;
        if (ProcMode)
          px[j++] = i;
        else if (c == symbol)
          flag = 1, j = i, pl = low;
      }
    }

    int Total = see_freq + low;

    if (ProcMode) {
      flag = count < low;
    }

    if (flag) {
      if (ProcMode) {
        for (low = 0, i = 0; (low += p[j = px[i]].Freq) <= count; i++);
      } else {
        low = pl;
      }
      p += j;

      if (see_freq > 2) psee2c->Summ -= see_freq;
      psee2c->update();

      FoundState = p;
      if (enable_learn) {
        p[0].Freq += 4;
        q.SummFreq += 4;
        if (p[0].Freq > MAX_FREQ)
          FoundState = rescale(q, OrderFall, FoundState);
      }
      RunLength = InitRL;
      EscCount++;

    } else {
      low = Total;
      NumMasked = cnum;
      psee2c->Summ += Total - see_freq;
    }
  }

  struct qsym {
    word sym;
    word freq;
    word total;

    void store(uint _sym, uint _freq, uint _total) {
      sym = _sym;
      freq = _freq;
      total = _total;
    }
  };

  qsym SQ[1024];
  uint SQ_ptr;

  uint sqp[256];  // symbol probs

  uint trF[256];  // binary tree, freqs
  uint trT[256];  // binary tree, totals

  void ConvertSQ(void) {
    uint i, c, j, b, freq, total, prob;
    uint cum = 0xFFFFFF00;  // base coef, add 1 to each to remove zero probs

    for (i = 0; i < 256; i++)
      sqp[i] = 0, trF[i] = 0, trT[i] = 0;  // init for all symbols

    for (i = 0; i < SQ_ptr; i++) {
      c = SQ[i].sym;
      freq = SQ[i].freq;
      total = SQ[i].total;
      prob = qword(qword(cum) * freq) / total;
      if (c < 256) {
        sqp[c] = prob + 1;
      } else {
        cum = prob;
      }
    }

    // build a binary tree with ppmd probs
    for (c = 0; c < 256; c++) {
      for (i = 8; i != 0; i--) {
        j = (256 + c) >> i;
        b = (c >> (i - 1)) & 1;
        if (b == 0) trF[j] += sqp[c];
        trT[j] += sqp[c];
      }
    }
  }

  void processBinSymbol_T(PPM_CONTEXT& q, int) {
    STATE& rs = q.oneState();
    int i = NS2BSIndx[suff(&q)->NumStats] + PrevSuccess + q.Flags +
            ((RunLength >> 26) & 0x20);
    word& bs = BinSumm[QTable[rs.Freq - 1]][i];
    BSumm = bs;

    SQ[SQ_ptr++].store(rs.Symbol, BSumm + BSumm, SCALE);
    SQ[SQ_ptr++].store(256, SCALE - BSumm - BSumm, SCALE);  // escape

    CharMask[rs.Symbol] = EscCount;
    NumMasked = 0;
  }

  // encode in unmasked (maxorder) context
  void processSymbol1_T(PPM_CONTEXT& q, int) {
    STATE* p = getStats(&q);

    int cnum = q.NumStats;
    int low = 0;
    int freq = 0;
    int total = q.SummFreq;
    int i;

    for (i = 0, low = 0; i <= cnum; i++) {
      freq = p[i].Freq;
      SQ[SQ_ptr++].store(p[i].Symbol, freq, total);
      low += freq;
    }

    if (q.iSuffix) PrefetchData(suff(&q));

    NumMasked = cnum;
    for (i = 0; i <= cnum; i++) CharMask[p[i].Symbol] = EscCount;

    SQ[SQ_ptr++].store(256, total - low, total);
  }

  // encode in masked context
  void processSymbol2_T(PPM_CONTEXT& q, int) {
    STATE* p = getStats(&q);

    int c;
    int low;
    int see_freq;
    int cnum = q.NumStats;

    SEE2_CONTEXT* psee2c;
    if (cnum != 0xFF) {
      psee2c = SEE2Cont[QTable[cnum + 3] - 4];
      psee2c += (q.SummFreq > 10 * (cnum + 1));
      psee2c += 2 * (2 * cnum < suff(&q)->NumStats + NumMasked) + q.Flags;
      see_freq = psee2c->getMean() + 1;
    } else {
      psee2c = &DummySEE2Cont;
      see_freq = 1;
    }

    int i;
    for (i = 0, low = 0; i <= cnum; i++) {
      c = p[i].Symbol;
      if (CharMask[c] != EscCount) low += p[i].Freq;
    }
    int Total = see_freq + low;

    for (i = 0; i <= cnum; i++) {
      c = p[i].Symbol;
      if (CharMask[c] != EscCount) {
        SQ[SQ_ptr++].store(c, p[i].Freq, Total);
        CharMask[c] = EscCount;
      }
    }

    SQ[SQ_ptr++].store(256, see_freq, Total);
    NumMasked = cnum;
  }

  uint cxt;  // bit context
  uint y;    // prev bit

  uint Init(uint MaxOrder, uint MMAX, uint CutOff, uint filesize) {
    _MaxOrder = MaxOrder;
    _CutOff = CutOff;
    _MMAX = MMAX;
    _filesize = filesize;

    PPMD_STARTUP();

    if (!StartSubAllocator(_MMAX)) return 1;

    StartModelRare();

    cxt = 0;
    y = 1;  // bit context

    return 0;
  }

  ~ppmd_Model() { StopSubAllocator(); }

  void ppmd_PrepareByte(void) {
    SQ_ptr = 0;
    NumMasked = 0;
    int _OrderFall = OrderFall;

    PPM_CONTEXT* MinContext = MaxContext;
    if (MinContext->NumStats) {
      processSymbol1_T(MinContext[0], 0);
    } else {
      processBinSymbol_T(MinContext[0], 0);
    }

    while (1) {
      do {
        if (!MinContext->iSuffix) goto Break;
        OrderFall++;
        MinContext = suff(MinContext);
      } while (MinContext->NumStats == NumMasked);
      processSymbol2_T(MinContext[0], 0);
    }

  Break:
    EscCount++;
    NumMasked = 0;
    OrderFall = _OrderFall;

    ConvertSQ();
  }

  void ppmd_UpdateByte(uint c) {
    PPM_CONTEXT* MinContext = MaxContext;
    if (MinContext->NumStats) {
      processSymbol1<0>(MinContext[0], c);
    } else {
      processBinSymbol<0>(MinContext[0], c);
    }

    while (!FoundState) {
      do {
        OrderFall++;
        MinContext = suff(MinContext);
      } while (MinContext->NumStats == NumMasked);
      processSymbol2<0>(MinContext[0], c);
    }

    PPM_CONTEXT* p;
    if ((OrderFall != 0) || ((byte*)getSucc(FoundState) < UnitsStart)) {
      p = UpdateModel(MinContext);
      if (p) MaxContext = p;
    } else {
      p = MaxContext = getSucc(FoundState);
    }

    if (p == 0) {
      if (_CutOff) {
        RestoreModelRare();
      } else {
        StartModelRare();
      }
    }
  }

  void WriteToDisk(std::ofstream* s) {
    for (int i = 0; i < N_INDEXES + 1; ++i) {
      Serialize(s, BList[i]);
    }
    Serialize(s, GlueCount);
    Serialize(s, GlueCount1);
    Serialize(s, SubAllocatorSize);
    unsigned long long offset = pText - HeapStart;
    Serialize(s, offset);
    offset = UnitsStart - HeapStart;
    Serialize(s, offset);
    offset = LoUnit - HeapStart;
    Serialize(s, offset);
    offset = HiUnit - HeapStart;
    Serialize(s, offset);
    offset = AuxUnit - HeapStart;
    Serialize(s, offset);
    offset = (byte*)FoundState - HeapStart;
    Serialize(s, offset);
    offset = (byte*)MaxContext - HeapStart;
    Serialize(s, offset);
    offset = (byte*)saved_pc - HeapStart;
    Serialize(s, offset);
    Serialize(s, OrderFall);
    Serialize(s, EscCount);
    for (int i = 0; i < 256; ++i) {
      Serialize(s, CharMask[i]);
    }
    Serialize(s, BSumm);
    Serialize(s, RunLength);
    Serialize(s, InitRL);
    Serialize(s, NumMasked);
    Serialize(s, PrevSuccess);
    for (int i = 0; i < 25; ++i) {
      for (int j = 0; j < 64; ++j) {
        Serialize(s, BinSumm[i][j]);
      }
    }
    for (int i = 0; i < 23; ++i) {
      for (int j = 0; j < 32; ++j) {
        Serialize(s, SEE2Cont[i][j].Count);
        Serialize(s, SEE2Cont[i][j].Shift);
        Serialize(s, SEE2Cont[i][j].Summ);
      }
    }
    Serialize(s, DummySEE2Cont.Count);
    Serialize(s, DummySEE2Cont.Shift);
    Serialize(s, DummySEE2Cont.Summ);
    for (int i = 0; i < 1024; ++i) {
      Serialize(s, SQ[i].freq);
      Serialize(s, SQ[i].sym);
      Serialize(s, SQ[i].total);
    }
    Serialize(s, SQ_ptr);
    for (int i = 0; i < 256; ++i) {
      Serialize(s, sqp[i]);
      Serialize(s, trF[i]);
      Serialize(s, trT[i]);
    }
    Serialize(s, cxt);
    Serialize(s, y);
    Serialize(s, enable_learn);
  }

  void ReadFromDisk(std::ifstream* s) {
    for (int i = 0; i < N_INDEXES + 1; ++i) {
      Serialize(s, BList[i]);
    }
    Serialize(s, GlueCount);
    Serialize(s, GlueCount1);
    Serialize(s, SubAllocatorSize);
    unsigned long long offset;
    Serialize(s, offset);
    pText = HeapStart + offset;
    Serialize(s, offset);
    UnitsStart = HeapStart + offset;
    Serialize(s, offset);
    LoUnit = HeapStart + offset;
    Serialize(s, offset);
    HiUnit = HeapStart + offset;
    Serialize(s, offset);
    AuxUnit = HeapStart + offset;
    Serialize(s, offset);
    FoundState = (STATE*)(HeapStart + offset);
    Serialize(s, offset);
    MaxContext = (PPM_CONTEXT*)(HeapStart + offset);
    Serialize(s, offset);
    saved_pc = (PPM_CONTEXT*)(HeapStart + offset);
    Serialize(s, OrderFall);
    Serialize(s, EscCount);
    for (int i = 0; i < 256; ++i) {
      Serialize(s, CharMask[i]);
    }
    Serialize(s, BSumm);
    Serialize(s, RunLength);
    Serialize(s, InitRL);
    Serialize(s, NumMasked);
    Serialize(s, PrevSuccess);
    for (int i = 0; i < 25; ++i) {
      for (int j = 0; j < 64; ++j) {
        Serialize(s, BinSumm[i][j]);
      }
    }
    for (int i = 0; i < 23; ++i) {
      for (int j = 0; j < 32; ++j) {
        Serialize(s, SEE2Cont[i][j].Count);
        Serialize(s, SEE2Cont[i][j].Shift);
        Serialize(s, SEE2Cont[i][j].Summ);
      }
    }
    Serialize(s, DummySEE2Cont.Count);
    Serialize(s, DummySEE2Cont.Shift);
    Serialize(s, DummySEE2Cont.Summ);
    for (int i = 0; i < 1024; ++i) {
      Serialize(s, SQ[i].freq);
      Serialize(s, SQ[i].sym);
      Serialize(s, SQ[i].total);
    }
    Serialize(s, SQ_ptr);
    for (int i = 0; i < 256; ++i) {
      Serialize(s, sqp[i]);
      Serialize(s, trF[i]);
      Serialize(s, trT[i]);
    }
    Serialize(s, cxt);
    Serialize(s, y);
    Serialize(s, enable_learn);
  }

  void Copy(const MemoryInterface* m) {
    const ppmd_Model* orig = static_cast<const ppmd_Model*>(m);
    for (int i = 0; i < N_INDEXES + 1; ++i) {
      BList[i] = orig->BList[i];
    }
    GlueCount = orig->GlueCount;
    GlueCount1 = orig->GlueCount1;
    SubAllocatorSize = orig->SubAllocatorSize;
    pText = HeapStart + (orig->pText - orig->HeapStart);
    UnitsStart = HeapStart + (orig->UnitsStart - orig->HeapStart);
    LoUnit = HeapStart + (orig->LoUnit - orig->HeapStart);
    HiUnit = HeapStart + (orig->HiUnit - orig->HeapStart);
    AuxUnit = HeapStart + (orig->AuxUnit - orig->HeapStart);
    FoundState =
        (STATE*)(HeapStart + ((byte*)orig->FoundState - orig->HeapStart));
    MaxContext =
        (PPM_CONTEXT*)(HeapStart + ((byte*)orig->MaxContext - orig->HeapStart));
    saved_pc =
        (PPM_CONTEXT*)(HeapStart + ((byte*)orig->saved_pc - orig->HeapStart));
    OrderFall = orig->OrderFall;
    EscCount = orig->EscCount;
    for (int i = 0; i < 256; ++i) {
      CharMask[i] = orig->CharMask[i];
    }
    BSumm = orig->BSumm;
    RunLength = orig->RunLength;
    InitRL = orig->InitRL;
    NumMasked = orig->NumMasked;
    PrevSuccess = orig->PrevSuccess;
    for (int i = 0; i < 25; ++i) {
      for (int j = 0; j < 64; ++j) {
        BinSumm[i][j] = orig->BinSumm[i][j];
      }
    }
    for (int i = 0; i < 23; ++i) {
      for (int j = 0; j < 32; ++j) {
        SEE2Cont[i][j].Count = orig->SEE2Cont[i][j].Count;
        SEE2Cont[i][j].Shift = orig->SEE2Cont[i][j].Shift;
        SEE2Cont[i][j].Summ = orig->SEE2Cont[i][j].Summ;
      }
    }
    DummySEE2Cont.Count = orig->DummySEE2Cont.Count;
    DummySEE2Cont.Shift = orig->DummySEE2Cont.Shift;
    DummySEE2Cont.Summ = orig->DummySEE2Cont.Summ;
    for (int i = 0; i < 1024; ++i) {
      SQ[i].freq = orig->SQ[i].freq;
      SQ[i].sym = orig->SQ[i].sym;
      SQ[i].total = orig->SQ[i].total;
    }
    SQ_ptr = orig->SQ_ptr;
    for (int i = 0; i < 256; ++i) {
      sqp[i] = orig->sqp[i];
      trF[i] = orig->trF[i];
      trT[i] = orig->trT[i];
    }
    cxt = orig->cxt;
    y = orig->y;
    enable_learn = orig->enable_learn;
  }
};

#pragma pack()

ModPPMD::ModPPMD(ShortTermMemory& short_term_memory,
                 LongTermMemory& long_term_memory, int order, int memory)
    : top_(255), mid_(127), bot_(0) {
  prediction_index_ = short_term_memory.num_predictions++;
  ppmd_model_.reset(new ppmd_Model());
  ppmd_model_->Init(order, memory, 1, 0);
  long_term_memory.ppmd_memory = ppmd_model_->HeapStart;
  long_term_memory.ppmd_memory_size = ppmd_model_->SubAllocatorSize;
}

void ModPPMD::Predict(ShortTermMemory& short_term_memory,
                      const LongTermMemory& long_term_memory) {
  if (short_term_memory.recent_bits == 1) {
    // A new byte has been observed. Update the byte-level predictions.
    ppmd_model_->ppmd_UpdateByte(short_term_memory.last_byte);
    ppmd_model_->ppmd_PrepareByte();
    for (int i = 0; i < 256; ++i) {
      short_term_memory.ppm_predictions[i] = ppmd_model_->sqp[i];
      if (short_term_memory.ppm_predictions[i] < 1)
        short_term_memory.ppm_predictions[i] = 1;
    }
    short_term_memory.ppm_predictions /=
        short_term_memory.ppm_predictions.sum();
    top_ = 255;
    bot_ = 0;
  } else {
    if (short_term_memory.new_bit) {
      bot_ = mid_ + 1;
    } else {
      top_ = mid_;
    }
  }
  ppmd_model_->enable_learn = false;
  mid_ = bot_ + ((top_ - bot_) / 2);
  float num =
      std::accumulate(&short_term_memory.ppm_predictions[mid_ + 1],
                      &short_term_memory.ppm_predictions[top_ + 1], 0.0f);
  float denom =
      std::accumulate(&short_term_memory.ppm_predictions[bot_],
                      &short_term_memory.ppm_predictions[mid_ + 1], num);
  float p = 0.5;
  if (denom != 0) p = num / denom;
  short_term_memory.SetPrediction(p, prediction_index_);
}

void ModPPMD::Learn(const ShortTermMemory& short_term_memory,
                    LongTermMemory& long_term_memory) {
  ppmd_model_->enable_learn = true;
}

void ModPPMD::WriteToDisk(std::ofstream* s) {
  Serialize(s, top_);
  Serialize(s, mid_);
  Serialize(s, bot_);
  ppmd_model_->WriteToDisk(s);
}

void ModPPMD::ReadFromDisk(std::ifstream* s) {
  Serialize(s, top_);
  Serialize(s, mid_);
  Serialize(s, bot_);
  ppmd_model_->ReadFromDisk(s);
}

void ModPPMD::Copy(const MemoryInterface* m) {
  const ModPPMD* orig = static_cast<const ModPPMD*>(m);
  top_ = orig->top_;
  mid_ = orig->mid_;
  bot_ = orig->bot_;
  ppmd_model_->Copy(orig->ppmd_model_.get());
}

}  // namespace PPMD

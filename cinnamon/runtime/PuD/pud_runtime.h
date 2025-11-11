#ifndef PUD_RUNTIME_H
#define PUD_RUNTIME_H

#include <stdlib.h>

extern void rowop_ap(void *dst);
extern void rowop_aap(void *dst, void *src);

#define COL_NUM 8192
#define ALIGNMENT COL_NUM
// #define NUM_RANK_PER_CHANNEL 2
// #define NUM_BANK_PER_RANK 8
// #define NUM_SUBARRAY_PER_BANK 64
// #define NUM_ROW_PER_SUBARRAY 1024
// #define NUM_ROW_PER_BANK (NUM_ROW_PER_SUBARRAY * NUM_SUBARRAY_PER_BANK)
// #define NUM_ROW_PER_RANK (NUM_ROW_PER_BANK * NUM_BANK_PER_RANK)
// #define NUM_ROW_PER_CHANEL (NUM_ROW_PER_RANK * NUM_RANK_PER_CHANNEL)

// static void *B0  = NULL;
// static void *B1  = NULL;
// static void *B2  = NULL;
// static void *B3  = NULL;
// static void *B4  = NULL;
// static void *B5  = NULL;
// static void *B6  = NULL;
// static void *B7  = NULL;
// static void *B8  = NULL;
// static void *B9  = NULL;
// static void *B10 = NULL;
// static void *B11 = NULL;
// static void *B12 = NULL;
// static void *B13 = NULL;
// static void *B14 = NULL;
// static void *B15 = NULL;
// static void *C0  = NULL;
// static void *C1  = NULL;

// void *pud_get_row();
// void pud_aap(void *addr1, void *addr2);
// void pud_ap(void *addr);

static void *pud_get_row() {
  void *row_ptr;
  int dummy = posix_memalign(&row_ptr, ALIGNMENT, ALIGNMENT);
  if (dummy) exit(-1);
  return row_ptr;
}

static void pud_aap(void *addr1, void *addr2) {
  rowop_aap(addr2, addr1);
}

static void pud_ap(void *addr) {
  rowop_ap(addr);
}

#endif

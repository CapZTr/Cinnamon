#include "pud_runtime.h"
#include <stdlib.h>

#define COL_NUM 8192
#define ALIGNMENT COL_NUM

void *pud_get_row() {
  void *row_ptr;
  int dummy = posix_memalign(&row_ptr, ALIGNMENT, ALIGNMENT);
  if (dummy) exit(-1);
  return row_ptr;
}

void pud_aap(void *addr1, void *addr2) {
  rowop_aap(addr2, addr1);
}

void pud_ap(void *addr) {
  rowop_ap(addr);
}

// #include "pud_runtime.h"

// #include <cassert>
// #include <cstdlib>

// // void pud_init() {
// //   int dummy = 0;
// //   dummy += posix_memalign(&B0 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B1 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B2 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B3 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B4 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B5 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B6 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B7 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B8 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B9 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B10, ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B11, ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B12, ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B13, ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B14, ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&B15, ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&C0 , ALIGNMENT, ALIGNMENT);
// //   dummy += posix_memalign(&C1 , ALIGNMENT, ALIGNMENT);

// //   if (dummy) {
// //     std::cout << "Init failed\n";
// //     exit(-1);
// //   }
// // }

// void *pud_get_row() {
//   void *row_ptr;
//   int dummy = posix_memalign(&row_ptr, ALIGNMENT, ALIGNMENT);
//   if (dummy) exit(-1);
//   return row_ptr;
// }

// void pud_aap(void *addr1, void *addr2) {
//   rowop_aap(addr2, addr1);
// }

// void pud_ap(void *addr) {
//   rowop_ap(addr);
// }

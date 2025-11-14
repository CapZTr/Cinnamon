#ifndef PUD_RUNTIME_H
#define PUD_RUNTIME_H

extern void rowop_ap(void *dst);
extern void rowop_aap(void *dst, void *src);

void *pud_get_row();
void pud_aap(void *addr1, void *addr2);
void pud_ap(void *addr);

#endif

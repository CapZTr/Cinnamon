module {
  func.func @main(%arg0: !pud.row_addr<D, 10>, %arg1: !pud.row_addr<B, 1>, %arg2: !pud.row_addr<C, 0>) {
    pud.AAP %arg0, %arg1 : !pud.row_addr<D, 10>, !pud.row_addr<B, 1>
    pud.AP %arg2 : !pud.row_addr<C, 0>

    return
  }
}

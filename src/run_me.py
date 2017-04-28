import pre_processor as pp
import oneVSAllSVM as OVA
import oneVSOneSVM as OVO

pp.PreProcessor()

ova = OVA.OneVSAllSVM()
ova.run()

ovo = OVO.OneVSOneSVM()
ovo.run()





import pre_processor as pp
#import oneVSAllSVM as OVA
#import oneVSOneSVM as OVO


# import oneVSAllSVM as OVA
# import oneVSOne as OVO

processor = pp.PreProcessor()
processor.condense()
processor.extract_stats()
processor.condense_3d(60)
# ova = OVA.OneVSAllSVM()
# ova.run()
#
# ovo = OVO.OneVSOneSVM()
# ovo.run()
#
#
#
#

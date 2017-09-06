from .align import AlignDatabase, GetAlignFuncByBoundingBox, GetAlignFuncByLandmarks

if __name__ == '__main__':
    source = 'F:\\FV_TMP\\Data\\Raw'
    target = 'F:\\FV_TMP\\Data\\gray_182'

    F_te = GetAlignFuncByLandmarks(output_size=128, ec_y=40)
    F_tr = GetAlignFuncByLandmarks(output_size=144, ec_y=48)
    
    F_tmp = GetAlignFuncByLandmarks(output_size=182, ec_y=60)
    AlignDatabase(source, target, align_func=F_tmp)
        
    # F = GetAlignFuncByBoundingBox(image_size=128, margin=24)
    # AlignDatabase(source, target, align_func=F)
    
    
    
    
    
    
    
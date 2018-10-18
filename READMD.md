Step 1. Make a data set
    
    A.  Open helps\datamaker.py
    B.  Check the parameters in it:
        
        n_process = 10                                              # number of processes to use
        in_path = 'D:\\datasets\\all_raw_ocr'                       # folder that stores B1, B2, B3, ...
        out_imgs_path = 'D:\\datasets\\invoices\\imgs'              # folder for pure images
        out_debug_imgs_path = 'D:\\datasets\\invoices\\debug_imgs'  # folder for images with boxes on them
        out_masks_path = 'D:\\datasets\\invoices\\masks'            # folder for npy files of masks 
                                                                    # (dtype: np.bool, value: 0, 1)
        out_img_height = 1024                                       # output image height
        out_img_width = 768                                         # output image width
        distance_threshold = 40                                     # if the horizontal distance between
                                                                    # two boxes is smaller than it, they
                                                                    # are merged to be a larger box
        discard_threshold_for_width = 40                            # if a box has a width smaller than
                                                                    # it, it is discarded
        debug = True                                                # whether or not to store images with 
                                                                    # boxes on them
    C.  Run it.
    
    Note:   out_imgs_path and out_masks_path should in the same folder. The upper folder of out_imgs_path 
            and out_masks_path will be the data set folder, which is 'D:\\datasets\\invoices' in the above
            case.
    
    Note:   On 10/18/2018, the max number of boxes in all images is 408.

Step 2. Train a model

    A.  Open main_for_training.py
    B.  Modify the parameters in InvoiceConfig if necessary
    C.  Use make_datasets to make training data set and valdiate data set using the data set folder
    D.  Load pretrained weights for COCO data set
    E.  First of all, train the heads, i.e. RPN classifier, RPN box regressor, MRCNN classifier, 
        MRCNN box regressor and MRCNN mask regressor
    F.  Then train all of the network

Step 3. Evaluate the model

    A.  Open main_for_investigating.py
    B.  Load the latest checkpoint of a model
    C.  Detect boxes in a sample in the validate data set
    
    Note:   Use the same random seed of Numpy and thus we can have consistent training-validate splits of 
            the whole data in main_for_training.py and main_for_investigating.py.
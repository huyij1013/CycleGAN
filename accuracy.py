import os
from skimage import io
import numpy as np
groundtruth_path = r'E:\xview2\cyclegan_formal\fully_match_post\groundtruth'
google_path = r'C:\Users\84471\Desktop\image_all_predict_target'
target_path = r'E:\xview2\train\split_disaster\hurricane-harvey\masks2'
with open("eval_cyclegan_partial16.txt", "a") as evalfile:
    evalfile.write('file   iou   precision   recall   acc\n')
IOU = []
PRECISION = []
RECALL = []
ACC = []
for groundtruth,google in zip(os.listdir(groundtruth_path),os.listdir(google_path)):
    ref = io.imread(os.path.join(groundtruth_path,groundtruth))[:,:,0].repeat(2, axis=0).repeat(2, axis=1)
    pred = io.imread(os.path.join(google_path, google))

    '''只评价未损毁的建筑'''
    # target = io.imread(os.path.join(target_path,groundtruth.replace('pre','post')))[:,:,0].repeat(2, axis=0).repeat(2, axis=1)
    # if np.sum(target==1)!=0:
    #     ref = ref*(target==1)
    #     pred = pred*(target==1)

    ref[ref==255]=1
    pred[pred==255]=1

    inters = ref & pred
    union = ref | pred
    correct = ref == pred
    correct_building = (ref == pred)&(ref == 1)

    inters_count = np.count_nonzero(inters)
    union_count = np.count_nonzero(union)
    correct_count = np.count_nonzero(correct)
    correct_count_building = np.count_nonzero(correct_building)
    total_count = ref.size
    ref_building_count = np.sum(ref == 1)
    pred_building_count = np.sum(pred == 1)


    iou = inters_count / float(union_count)
    acc = correct_count / float(total_count)
    recall = correct_count_building / float(ref_building_count)
    if  float(pred_building_count)!=0:
        precision = correct_count_building/float(pred_building_count)
    else:
        precision = 0
        print(groundtruth)
    IOU.append(iou)
    ACC.append(acc)
    RECALL.append(recall)
    PRECISION.append(precision)
    with open("eval_cyclegan_partial16.txt", "a") as evalfile:
        evalfile.write(groundtruth)
        evalfile.write('  ')
        evalfile.write(str(iou))
        evalfile.write('  ')
        evalfile.write(str(precision))
        evalfile.write('  ')
        evalfile.write(str(recall))
        evalfile.write('  ')
        evalfile.write(str(acc)+'\n')
with open("eval_cyclegan_partial16.txt", "a") as evalfile:
    evalfile.write(str(np.mean(np.array(IOU))))
    evalfile.write('  ')
    evalfile.write(str(np.mean(np.array(PRECISION))))
    evalfile.write('  ')
    evalfile.write(str(np.mean(np.array(RECALL))))
    evalfile.write('  ')
    evalfile.write(str(np.mean(np.array(ACC))) + '\n')
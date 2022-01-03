import torch
import numpy as np
import pandas as pd
import torchvision

RESIZE_WIDTH = 448
RESIZE_HEIGHT = 448


class DecodeBB:
    
    def __init__(self, width=RESIZE_WIDTH,
                       height=RESIZE_HEIGHT,
                       n_classes=20, 
                       n_segments=7, 
                       min_confidence=0.5,
                       iou_thresh=0.5,
                       ):
        
        super().__init__()
        self.width = width
        self.height = height

        
        self.n_classes = n_classes
        self.n_segments = n_segments
        self.min_confidence = min_confidence
        self.iou_thresh = iou_thresh
        
        self.init_centroid_ref()
        
        
    def init_centroid_ref(self):
                
        self.xc_ref = np.vstack(self.n_segments * [np.arange(self.n_segments)])        
        self.yc_ref = np.vstack(self.n_segments * [np.arange(self.n_segments)]).T
        
        self.xc_ref = self.xc_ref * self.width / self.n_segments
        self.yc_ref = self.yc_ref * self.height / self.n_segments
        
        self.xc_del = (self.width / self.n_segments)
        self.yc_del = (self.height / self.n_segments)
        

        
    def get_centers(self, bb_matrix):
                
        x_offset_01 = bb_matrix[:,self.n_classes + 0,:,:]
        y_offset_01 = bb_matrix[:,self.n_classes + 1,:,:]
        
        x_offset_02 = bb_matrix[:,self.n_classes + 5,:,:]
        y_offset_02 = bb_matrix[:,self.n_classes + 6,:,:]
        
        batch_size = bb_matrix.shape[0]
        
        xc_ref_batch = np.stack(batch_size * [self.xc_ref])
        yc_ref_batch = np.stack(batch_size * [self.yc_ref])
        
        xc1 = xc_ref_batch + x_offset_01 * self.xc_del
        yc1 = yc_ref_batch + y_offset_01 * self.yc_del
        
        xc2 = xc_ref_batch + x_offset_02 * self.xc_del
        yc2 = yc_ref_batch + y_offset_02 * self.yc_del
        
        return xc1, yc1, xc2, yc2
    
    
    def get_dims(self, bb_matrix):
                
        return (
            bb_matrix[:,self.n_classes + 2,:,:] * self.width,
            bb_matrix[:,self.n_classes + 3,:,:] * self.height,
            bb_matrix[:,self.n_classes + 2 + 5,:,:] * self.width,
            bb_matrix[:,self.n_classes + 3 + 5,:,:] * self.height
        )
    
    def get_confidence(self, bb_matrix):
        return bb_matrix[:,24,:,:], bb_matrix[:,29,:,:]
    
    
    def nms_per_class(self, bb_data):
        bb_data = bb_data[bb_data['confidence'] > self.min_confidence]
        bb_data = bb_data.reset_index(drop=True)
        
        res = []
        
        for class_idx in range(0, self.n_classes):
            
            bb_sdata = bb_data[bb_data['class_idx'] == class_idx].reset_index(drop=True)

    #         bb_sdata = bb_data

            if bb_sdata.shape[0] > 0:
                bboxes = bb_sdata[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
                bboxes = torch.Tensor(bboxes)

                scores = bb_sdata['confidence'].to_numpy()
                scores = torch.Tensor(scores)

                nms_res = torchvision.ops.nms(bboxes, scores, self.iou_thresh)
                nms_idx = nms_res[0].numpy()

                res.append({
                    'xmin' : bb_sdata['xmin'][nms_idx],
                    'ymin' : bb_sdata['ymin'][nms_idx],
                    'xmax' : bb_sdata['xmax'][nms_idx],
                    'ymax' : bb_sdata['ymax'][nms_idx],
                    'class_idx' : bb_sdata['class_idx'][nms_idx],
                    'class_prob' : bb_sdata['class_prob'][nms_idx]
                })

        return res
                
    
        
    def compile_bbox(self, bb_matrix):
        
        batch_size = bb_matrix.shape[0]
        
        xc1, yc1, xc2, yc2 = self.get_centers(bb_matrix)
        w1, h1, w2, h2 = self.get_dims(bb_matrix)
        c1, c2 = self.get_confidence(bb_matrix)
        
        batch_bbs = []
        batch_scores = []
        batch_class_prob = []
                    
        xmax1 = xc1 + 0.5 * w1
        xmin1 = xmax1 - w1
            
        ymax1 = yc1 + 0.5 * h1
        ymin1 = ymax1 - h1
        
        xmax2 = xc2 + 0.5 * w2
        xmin2 = xmax2 - w2
            
        ymax2 = yc2 + 0.5 * h2
        ymin2 = ymax2 - h2
        
        batch_result = []
        
        for batch_idx in range(batch_size):

            xmin = np.hstack([xmin1[batch_idx,:,:], xmin2[batch_idx,:,:]]).reshape(-1,1)
            xmax = np.hstack([xmax1[batch_idx,:,:], xmax2[batch_idx,:,:]]).reshape(-1,1)
            
            ymin = np.hstack([ymin1[batch_idx,:,:], ymin2[batch_idx,:,:]]).reshape(-1,1) 
            ymax = np.hstack([ymax1[batch_idx,:,:], ymax2[batch_idx,:,:]]).reshape(-1,1)
            
            conf = np.hstack([c1[batch_idx,:,:], c2[batch_idx,:,:]]).reshape(-1,1)
        
            class_idx = np.argmax(bb_matrix[batch_idx,0:20,:,:], axis=0)
            class_idx = np.hstack([class_idx, class_idx]).reshape(-1,1)
            
            class_prob = np.max(bb_matrix[batch_idx,0:20,:,:], axis=0)
            class_prob = np.hstack([class_prob, class_prob]).reshape(-1,1)
            
            bbox_data = np.hstack([
                xmin, 
                ymin, 
                xmax,
                ymax, 
                conf, 
                class_idx, 
                class_prob
            ])
            
            bbox_data = pd.DataFrame(bbox_data, columns=[
                'xmin', 'ymin', 'xmax', 'ymax', 'confidence',
                'class_idx', 'class_prob'
            ])
            
            batch_result.append(self.nms_per_class(bbox_data))
            
        return batch_result
            

        
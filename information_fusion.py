import math
import numpy as np

# LARA.py use this file to calculate risk index and fuse the model's predictions 
# Information fusion: Technique to combine multiple sources of data or predictions into a single, more comprehensive result 
# Refine & combine model predictions for a more reliable and interpretable risk analysis 
# Aggregates overlapping CNN predictions (y_hat) for each 1-minute unit of fetal heart monitoring, using different fusion strategies (Basic, R-S, C-W)
def information_fusion(y_hat, cam_value, use_weight,use_cam):
    monitor_time=len(cam_value)+9
    infused_cam=cam_value.reshape(len(cam_value),10,240).mean(axis=2)
    infusion_result=np.zeros(shape=(monitor_time))

    for idx in range(monitor_time):
        # Normal case: central regions
        if idx>=9 and monitor_time-1-idx>=9:
            cam_weights=[infused_cam[idx-i,i] for i in range(9,-1,-1)]##这个是正着取的
            y=y_hat[idx-9:idx+1] 
            weights=[math.exp(yi-sum(y)/len(y)) for yi in y]
            assert len(weights)==10
        # Edge case: not enough earlier segments
        if idx<=8:
            cam_weights=[infused_cam[i,idx-i] for i in range(idx+1)]
            y=y_hat[0:idx+1]
            weights=[math.exp(yi-sum(y)/len(y)) for yi in y]
        
        # Edge case: Late minutes (adjust when nearing the last few minutes of the timeline)
        if monitor_time-1-idx<=8:
            cam_weights=[infused_cam[idx-i,i] for i in range(9,idx+9-monitor_time,-1)]
            y=y_hat[idx-9:]
            weights=[math.exp(yi-sum(y)/len(y)) for yi in y]
        
        # Apply fusion strategy
        if use_weight:
            if use_cam:
                #cam weighted operator (use model attention as weights)
                infusion_result[idx]=np.multiply(np.array(cam_weights),y).sum()/sum(cam_weights)
            else:
                #risk-sensitive operator (prioritize higher predicted risk)
                infusion_result[idx]=np.multiply(np.array(weights),y).sum()/sum(weights)
        else:
            #basic operator (simple avg of the 10 overlapping predictions)
            infusion_result[idx]=sum(y)/len(y)
    return infusion_result # Returns a minute-level risk trend across the entire monitoring period.
import torch


NUM_CLASSES = 20

XC_IDX_01 = 20
YC_IDX_01 = 21

XC_IDX_02 = 25
YC_IDX_02 = 26

WIDTH_IDX_01 = 22
HEIGHT_IDX_01 = 23

WIDTH_IDX_02 = 27
HEIGHT_IDX_02 = 28


OBJ_IDX_01 = 24
OBJ_IDX_02 = 29


LAMBDA_OBJ = 5.0
LAMBDA_NOOBJ = 0.5

NUM_SEGMENTS = 7


def loss_center(target, pred):

    xc_diff01 = (target[:,XC_IDX_01,:,:] - pred[:,XC_IDX_01,:,:]) ** 2
    yc_diff01 = (target[:,YC_IDX_01,:,:] - pred[:,YC_IDX_01,:,:]) ** 2

    xc_diff02 = (target[:,XC_IDX_02,:,:] - pred[:,XC_IDX_02,:,:]) ** 2
    yc_diff02 = (target[:,YC_IDX_02,:,:] - pred[:,YC_IDX_02,:,:]) ** 2

    obj01 = target[:,OBJ_IDX_01,:,:]
    obj02 = target[:,OBJ_IDX_02,:,:]

    sum_centers = obj01 * (xc_diff01 + yc_diff01) + obj02 * (xc_diff02 + yc_diff02)

    sum_centers = sum_centers.view(-1, 1 * NUM_SEGMENTS * NUM_SEGMENTS)
    
    return LAMBDA_OBJ * torch.mean(sum_centers, dim=1)


def loss_dimmensions(target, pred):

    width_diff01 = (target[:,WIDTH_IDX_01,:,:] ** 0.5 - pred[:,WIDTH_IDX_01,:,:] ** 0.5) ** 2
    height_diff01 = (target[:,HEIGHT_IDX_01,:,:] ** 0.5 - pred[:,HEIGHT_IDX_01,:,:] ** 0.5) ** 2

    width_diff02 = (target[:,WIDTH_IDX_02,:,:] ** 0.5 - pred[:,WIDTH_IDX_02,:,:] ** 0.5) ** 2
    height_diff02 = (target[:,HEIGHT_IDX_02,:,:] ** 0.5 - pred[:,HEIGHT_IDX_02,:,:] ** 0.5) ** 2

    obj01 = target[:,OBJ_IDX_01,:,:]
    obj02 = target[:,OBJ_IDX_02,:,:]

    sum_dims = obj01 * (width_diff01 + height_diff01) + obj02 * (width_diff02 + height_diff02)

    sum_dims = sum_dims.view(-1, NUM_SEGMENTS * NUM_SEGMENTS)
    
    return LAMBDA_OBJ * torch.mean(sum_dims, dim=1)


def loss_obj(target, pred):

    obj_diff_01 = (target[:,OBJ_IDX_01,:,:] - pred[:,OBJ_IDX_01,:,:]) ** 2
    obj_diff_02 = (target[:,OBJ_IDX_02,:,:] - pred[:,OBJ_IDX_02,:,:]) ** 2

    sum_objs = (obj_diff_01 + obj_diff_02).view(-1, NUM_SEGMENTS * NUM_SEGMENTS)

    return torch.mean(sum_objs, dim=1)


def loss_noobj(target, pred):
 
    target_noobj01 = (target[:,OBJ_IDX_01,:,:] - 1)
    target_noobj02 = (target[:,OBJ_IDX_02,:,:] - 1)

    pred_noobj01 = (pred[:,OBJ_IDX_01,:,:] - 1)
    pred_noobj02 = (pred[:,OBJ_IDX_02,:,:] - 1)

    noobj_diff_01 = (target_noobj01 - pred_noobj01) ** 2
    noobj_diff_02 = (target_noobj02 - pred_noobj02) ** 2

    sum_noobjs = (noobj_diff_01 + noobj_diff_02).view(-1, NUM_SEGMENTS * NUM_SEGMENTS)

    return LAMBDA_NOOBJ * torch.mean(sum_noobjs, dim=1)


def loss_class_prob(target, pred):

    diff = (target[:,0:NUM_CLASSES,:,:] - pred[:,0:NUM_CLASSES,:,:])**2
    diff = torch.sum(diff, dim=1)
    diff =  target[:,OBJ_IDX_01,:,:] * diff
    diff = diff.view(-1, NUM_SEGMENTS * NUM_SEGMENTS)

    return torch.mean(diff, dim=1) 


# def compute_loss(yt, yp):
#     loss_center_ = loss_center(yt, yp)
#     loss_dims_ = loss_dimmensions(yt, yp)
#     loss_obj_ = loss_obj(yt, yp)
#     loss_noobj_ = loss_noobj(yt, yp)
#     loss_class_prob_ = loss_class_prob(yt, yp)
    
#     loss = torch.mean(loss_center_ + loss_dims_ + loss_obj_ + loss_noobj_ + loss_class_prob_)
#     return loss









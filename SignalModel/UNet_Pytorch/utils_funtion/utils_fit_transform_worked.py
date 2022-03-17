from itertools import cycle

import torch
from tqdm import tqdm

from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss, CORAL
from funtion.utils import get_lr
from funtion.utils_metrics import f_score


def fit_one_epoch_transform(model_train, model, loss_history,
                            optimizer, epoch, epoch_step, epoch_step_val,
                            dataloads, Epoch, cuda, dice_loss, focal_loss,
                            cls_weights, num_classes, tfwriter, best_val_loss):
    total_loss          = 0
    dice_loss_item      = 0
    ce_loss_item        = 0
    coral_loss_item     = 0
    total_f_score       = 0

    val_ce_loss_item    = 0
    val_dice_loss_item  = 0
    val_loss            = 0
    val_coral_loss      = 0
    val_f_score         = 0

    source_gen, source_gen_val, target_gen, target_gen_val = dataloads

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, (source_batch, target_batch) in enumerate(zip(cycle(source_gen), target_gen)):
            if iteration >= epoch_step:
                break
            source_imgs, source_pngs, source_labels = source_batch
            target_imgs, _, _ = target_batch

            with torch.no_grad():
                source_imgs    = torch.from_numpy(source_imgs).type(torch.FloatTensor)
                source_pngs    = torch.from_numpy(source_pngs).long()
                source_labels  = torch.from_numpy(source_labels).type(torch.FloatTensor)
                target_imgs    = torch.from_numpy(target_imgs).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    source_imgs    = source_imgs.cuda()
                    source_pngs    = source_pngs.cuda()
                    source_labels  = source_labels.cuda()
                    target_imgs    = target_imgs.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            # 网络预测结果
            source_outputs, source_out = model_train(source_imgs)
            target_outputs, target_out = model_train(target_imgs)
            # 计算CE loss
            if focal_loss:
                loss = Focal_Loss(source_outputs, source_pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(source_outputs, source_pngs, weights, num_classes = num_classes)

            # 将celoss结果保存下来
            ce_loss_item    += loss.item()
            # 计算Dice loss
            if dice_loss:
                main_dice = Dice_loss(source_outputs, source_labels)
                loss      = loss + main_dice
                # 将diceloss结果保存下来
                dice_loss_item  += main_dice.item()

            # 计算CORAL loss
            coral_loss = CORAL(sourc_fe_outputs, target_outputs)
            loss = loss + coral_loss
            coral_loss_item += coral_loss.item()

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(source_outputs, source_labels)

            loss.backward()
            optimizer.step()

            total_loss      += loss.item()
            total_f_score   += _f_score.item()

            pbar.set_postfix(**{'total_loss' : total_loss / (iteration + 1),
                                'f_score'    : total_f_score / (iteration + 1),
                                'Dice'    : dice_loss_item / (iteration + 1),
                                'coral' : coral_loss_item / (iteration + 1),
                                'lr'         : get_lr(optimizer)})
            pbar.update(1)

            with torch.no_grad():
                tfwriter.add_scalar('train/DiceLoss',  ce_loss_item / (iteration + 1), (epoch + 1) * (iteration + 1))
                tfwriter.add_scalar('train/CELoss',    dice_loss_item/ (iteration + 1), (epoch + 1) * (iteration + 1))
                tfwriter.add_scalar('train/TotalLoss', total_loss / (iteration + 1), (epoch + 1) * (iteration + 1))
                tfwriter.add_scalar('train/f_score',   total_f_score / (iteration + 1), (epoch + 1) * (iteration + 1))
                tfwriter.add_scalar('train/coral_loss', coral_loss_item / (iteration + 1), (epoch + 1) * (iteration + 1))

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, (source_batch_val, target_batch_val) in enumerate(zip(cycle(source_gen_val), target_gen_val)):
            if iteration >= epoch_step_val:
                break
            source_imgs_val, source_pngs_val, source_labels_val = source_batch_val
            target_imgs_val, _, _ = target_batch_val

            with torch.no_grad():
                source_imgs_val    = torch.from_numpy(source_imgs_val).type(torch.FloatTensor)
                source_pngs_val    = torch.from_numpy(source_pngs_val).long()
                source_labels_val  = torch.from_numpy(source_labels_val).type(torch.FloatTensor)
                target_imgs_val    = torch.from_numpy(target_imgs_val).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    source_imgs_val    = source_imgs_val.cuda()
                    source_pngs_val    = source_pngs_val.cuda()
                    source_labels_val  = source_labels_val.cuda()
                    target_imgs_val    = target_imgs_val.cuda()
                    weights = weights.cuda()

                source_outputs, source_out = model_train(source_imgs_val)
                traget_outputs, traget_out = model_train(target_imgs_val)
                if focal_loss:
                    loss = Focal_Loss(source_outputs, source_pngs_val, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(source_outputs, source_pngs_val, weights, num_classes = num_classes)

                # 将celoss结果保存下来
                val_ce_loss_item += loss.item()

                if dice_loss:
                    main_dice = Dice_loss(source_outputs, source_labels_val)
                    loss  = loss + main_dice
                    # 将celoss结果保存下来
                    val_dice_loss_item += main_dice.item()

                # 计算CORAL loss
                coral_loss = CORAL(source_outputs, traget_outputs)
                loss = loss + coral_loss
                coral_loss_item += coral_loss.item()

                # 计算f_score
                _f_score    = f_score(source_outputs, source_labels_val)

                val_loss    += loss.item()
                val_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'coral' : coral_loss_item / (iteration + 1),
                                'Dice'    : dice_loss_item / (iteration + 1)})
            pbar.update(1)

            tfwriter.add_scalar('val/DiceLoss',  val_dice_loss_item / (iteration + 1), (epoch + 1) * (iteration + 1),)
            tfwriter.add_scalar('val/CELoss',    val_ce_loss_item / (iteration + 1)  , (epoch + 1) * (iteration + 1),)
            tfwriter.add_scalar('val/TotalLoss', val_loss / (iteration + 1)         , (epoch + 1) * (iteration + 1),)
            tfwriter.add_scalar('val/f_score',   val_f_score / (iteration + 1)      , (epoch + 1) * (iteration + 1),)
            tfwriter.add_scalar('train/coral_loss', coral_loss_item / (iteration + 1), (epoch + 1) * (iteration + 1))

    loss_history.append_loss(total_loss/(epoch_step+1), val_loss/(epoch_step_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))

    return (epoch + 1), total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes):
    total_loss      = 0
    total_f_score   = 0

    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs    = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs    = torch.from_numpy(pngs).long()
                labels  = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs    = imgs.cuda()
                    pngs    = pngs.cuda()
                    labels  = labels.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss      += loss.item()
            total_f_score   += _f_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    loss_history.append_loss(total_loss/(epoch_step+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f' % (total_loss / (epoch_step + 1)))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / (epoch_step + 1)))

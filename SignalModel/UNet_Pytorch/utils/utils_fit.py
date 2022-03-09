import torch
from SignalModel.UNet_Pytorch.nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from SignalModel.UNet_Pytorch.utils.utils import get_lr
from SignalModel.UNet_Pytorch.utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history,
                  optimizer, epoch, epoch_step, epoch_step_val,
                  gen, gen_val, Epoch, cuda, dice_loss, focal_loss,
                  cls_weights, num_classes, tfwriter, best_val_loss):
    total_loss          = 0
    dice_loss_item      = 0
    ce_loss_item        = 0
    total_f_score       = 0

    val_ce_loss_item    = 0
    val_dice_loss_item  = 0
    val_loss            = 0
    val_f_score         = 0

    model_train.train()
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

            # 将celoss结果保存下来
            ce_loss_item    += loss.item()

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice
                # 将diceloss结果保存下来
                dice_loss_item  += main_dice.item()

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

            with torch.no_grad():
                tfwriter.add_scalar('train/DiceLoss',  ce_loss_item / (iteration + 1), (epoch + 1) * (iteration + 1))
                tfwriter.add_scalar('train/CELoss',    dice_loss_item/ (iteration + 1), (epoch + 1) * (iteration + 1))
                tfwriter.add_scalar('train/TotalLoss', total_loss / (iteration + 1), (epoch + 1) * (iteration + 1))
                tfwriter.add_scalar('train/f_score',   total_f_score / (iteration + 1), (epoch + 1) * (iteration + 1))

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
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

                outputs     = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                # 将celoss结果保存下来
                val_ce_loss_item += loss.item()

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss  = loss + main_dice
                    # 将celoss结果保存下来
                    val_dice_loss_item += main_dice.item()

                # 计算f_score
                _f_score    = f_score(outputs, labels)

                val_loss    += loss.item()
                val_f_score += _f_score.item()
                
            
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

            tfwriter.add_scalar('val/DiceLoss',  val_dice_loss_item / (iteration + 1), (epoch + 1) * (iteration + 1),)
            tfwriter.add_scalar('val/CELoss',    val_ce_loss_item / (iteration + 1)  , (epoch + 1) * (iteration + 1),)
            tfwriter.add_scalar('val/TotalLoss', val_loss / (iteration + 1)         , (epoch + 1) * (iteration + 1),)
            tfwriter.add_scalar('val/f_score',   val_f_score / (iteration + 1)      , (epoch + 1) * (iteration + 1),)
            
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

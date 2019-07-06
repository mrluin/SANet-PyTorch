import torch

class Metrics(object):
    def __init__(self,
                 logits,
                 target,
                 nb_classes):
        super(Metrics, self).__init__()

        self.acc, self.f1_score, self.iou= self._get_stat(logits, target, nb_classes)

        self.mean_f1_score = self.f1_score[0:nb_classes-1].sum() / (nb_classes-1)
        self.mean_iou = self.iou[0:nb_classes-1].sum() / (nb_classes-1)

    def _get_stat(self, logits, target, nb_classes):

        with torch.no_grad():

            pred = torch.argmax(logits, dim=1).view(-1)
            target = target.view(-1)
            # cpu version
            pixel_counter = torch.zeros(nb_classes)
            acc = torch.zeros(nb_classes)
            f1 = torch.zeros(nb_classes)
            iou = torch.zeros(nb_classes)
            for k in range(0, nb_classes):

                # tp + fp
                pred_inds = pred == k
                #    tp + fn
                target_inds = target == k
                #    fn + tn
                non_pred_inds = pred != k
                #    fp + tn
                non_target_inds = target != k
                # tp
                interection = pred_inds[target_inds].long().sum().float()
                # tp + fn + fp
                union = pred_inds.long().sum().float() + target_inds.long().sum().float() - interection

                # tn
                non_interection = non_pred_inds[non_target_inds].long().sum().float()
                # fn + fp /  tp + fp + tp + fn
                #denominator = non_pred_inds.long().sum().float() + non_target_inds.long().sum().float() - 2*non_interection
                denominator = pred_inds.long().sum().float() + target_inds.long().sum().float()
                pixel_counter[k] = target_inds.long().sum().float()

                acc[k] = interection

                f1[k] = ((2 * interection) / (denominator + 1e-10))

                iou[k] = (interection / (union + 1e-10))

        return (acc[:nb_classes-1].sum() / (pixel_counter[:nb_classes-1].sum() + 1e-10)), f1, iou


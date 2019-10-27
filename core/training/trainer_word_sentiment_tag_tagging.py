from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
from core.nn.utils import prepare_input2 as prepare_input
from core.nn.utils import lr_decay
from core.nn.loss import SentimentLoss


class ModelTrainer(object):
    """
    Class that controls the training process.
    """

    def __init__(self, model, lr, clip, device):
        self.model = model
        self.clip = clip
        self.device = device
        self.lr = lr

        self.parameters = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = optim.Adam(self.parameters, lr)

        self.criterion = SentimentLoss(5)

    def train_epoch(self, data_loader, report_step=100):

        stop_next_epoch = False
        train_loss = 0.
        train_acc = 0.
        n_sample = 0.
        for step, (batch_x, batch_len, batch_y, batch_x_tag) in enumerate(data_loader):
            n_sample += len(batch_y)

            bx, bx_len, bx_mask, by, bx_tag = prepare_input(batch_x, batch_len, batch_y, batch_x_tag)

            batch_x = bx.to(self.device)
            batch_x_mask = bx_mask.to(self.device)
            batch_len = bx_len.to(self.device)
            batch_y = by.to(self.device)
            batch_x_tag = bx_tag.to(self.device)

            pred, tags_logits = self.model(batch_x, batch_x_tag, batch_len, batch_x_mask)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = self.criterion(tags_logits, batch_x_tag, pred, batch_y)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=self.clip)
            self.optimizer.step()

            train_loss += loss.item()
            y_p = pred.max(dim=1)[1]
            train_correct = (y_p == batch_y).sum().item()
            train_acc += train_correct

            if (step+1) % report_step == 0:
                train_loss /= n_sample
                train_acc /= n_sample
                if train_loss < 1.e-4:
                    stop_next_epoch = True
                print("[step_%05d] loss: %0.4f, acc: %0.4f" % (step+1, train_loss, train_acc))
                train_loss = 0.
                train_acc = 0.
                n_sample = 0.
        return stop_next_epoch

    def train(self, n_epoch, train_loader, dev_loader=None, test_loader=None):
        # best_dev_acc = 0.
        # best_dev_loss = 1e30
        # best_test_acc = 0.
        # best_test_loss = 1e30

        dev_acc_list = []
        test_acc_list = []

        # set model in training mode.
        self.model.train()

        for epoch_i in range(1, n_epoch + 1):
            print('epoch {}'.format(epoch_i))
            if epoch_i > 1:
                self.optimizer = lr_decay(self.optimizer, epoch_i, 0.03, self.lr)
            else:
                print("init lr: %f" % self.lr)

            stop = self.train_epoch(train_loader)

            if dev_loader is not None:
                dev_acc = ModelTrainer.test(dev_loader, self.model, self.device)
                dev_acc_list.append(dev_acc)
                print("Epoch_%05d: dev acc: %0.4f" % (epoch_i, dev_acc))

            if test_loader is not None:
                test_acc = ModelTrainer.test(test_loader, self.model, self.device)
                test_acc_list.append(test_acc)
                print("Epoch_%05d: test acc: %0.4f" % (epoch_i, test_acc))

            if stop:
                print("stop training.")
                break

        return dev_acc_list, test_acc_list

    @staticmethod
    def test(data_loader, model, device):
        # set model in validating mode.
        model.eval()
        # model.to(device)
        correct = 0
        n = 0
        for (batch_x, batch_len, batch_y, batch_x_tag) in data_loader:
            n += len(batch_y)
            bx, bx_len, bx_mask, by, bx_tag = prepare_input(batch_x, batch_len, batch_y, batch_x_tag)

            batch_x = bx.to(device)
            batch_len = bx_len.to(device)
            batch_y = by.to(device)
            batch_max = bx_mask.to(device)
            batch_x_tag = bx_tag.to(device)

            try:
                output, _ = model(batch_x, batch_x_tag, batch_len, batch_max)

                # get the index of the max log-probability
                _, y_p = output.max(1)
                correct += (y_p == batch_y).sum().item()
            except:
                print("in test ..... ")
                print(batch_x.shape)
                print(batch_len.shape)
                print(batch_y.shape)
                input()
                pass

        eval_acc = correct / n
        # set model back to training mode.
        model.train()

        return eval_acc

    @staticmethod
    def get_attention_scores(data_loader, model, device):
        # set model in validating mode.
        model.eval()
        # model.to(device)

        scores = []
        data_lens = []
        n = 0
        for (batch_x, batch_len, batch_y, batch_x_tag) in data_loader:
            n += len(batch_y)
            bx, bx_len, bx_mask, by, bx_tag = prepare_input(batch_x, batch_len, batch_y, batch_x_tag)

            batch_x = bx.to(device)
            batch_len = bx_len.to(device)
            batch_max = bx_mask.to(device)

            batch_scores = model.get_attention(batch_x, batch_len, batch_max)
            batch_scores = batch_scores.cpu().detach().numpy()
            scores.append(batch_scores)
            data_lens.append(bx_len)

        return scores, data_lens

    @staticmethod
    def save(model, path):
        torch.save(model, path)

    @staticmethod
    def load(path):
        return torch.load(path)

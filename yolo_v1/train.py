import torch 
import loss_utils
import torch.optim as optim
import numpy as np
import loss_utils


def compute_loss(yt, yp):
    loss_center = loss_utils.loss_center(yt, yp)
    loss_dims = loss_utils.loss_dimmensions(yt, yp)
    loss_obj = loss_utils.loss_obj(yt, yp)
    loss_noobj = loss_utils.loss_noobj(yt, yp)
    loss_class_prob = loss_utils.loss_class_prob(yt, yp)
    loss = torch.mean(loss_center + loss_dims + loss_obj + loss_noobj + loss_class_prob)
    return loss



def train_model(model, tr_dl, val_dl, device, num_epoch, model_path, patience=5):

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-6)
    
    total_tr = len(tr_dl)
    total_val = len(val_dl)

    tr_loss_epoch = []
    val_loss_epoch = []
    
    cur_min_val_loss = 99999.9
    
    patience_counter = 0
    
    
    for epoch in range(num_epoch):

        tr_run_loss = 0.0
        tr_mean_loss = 0.0

        for cnt, data in enumerate(tr_dl, start=1):

            X,yt = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            yp = model(X)

            tr_loss = compute_loss(yt, yp)

            tr_loss.backward()
            optimizer.step()

            tr_run_loss += np.mean(tr_loss.cpu().detach().numpy())
            tr_mean_loss = tr_run_loss / cnt

            print(f'Epoch {epoch + 1} [{cnt}/{total_tr}] Training loss = {tr_mean_loss}', end='\r')

        tr_loss_epoch.append(tr_mean_loss)

        print('')


        with torch.no_grad():

            val_run_loss = 0.0
            val_mean_loss = 0.0

            for cnt, data in enumerate(val_dl, start=1):

                X,yt = data[0].to(device), data[1].to(device)
                yp = model(X)

                val_loss = compute_loss(yt, yp)
                val_run_loss += np.mean(val_loss.cpu().detach().numpy())
                val_mean_loss = val_run_loss / cnt


                print(f'Epoch {epoch + 1} [{cnt}/{total_tr}] Validation loss = {val_mean_loss}', end='\r')

            val_loss_epoch.append(val_mean_loss)

            print('')

            if val_mean_loss < cur_min_val_loss:
                cur_min_val_loss = val_mean_loss
                patience_counter = 0
                torch.save(model, model_path)
                print('Patience resets to ', patience_counter)
                print(f'Updating best Model saved to {model_path}')
            else:
                patience_counter += 1
                print(f'Patience Counter = {patience_counter}')


        if patience_counter >= patience:
            print('Exceeded Patience... Stopping Training')
            break    
    
    return tr_loss_epoch, val_loss_epoch





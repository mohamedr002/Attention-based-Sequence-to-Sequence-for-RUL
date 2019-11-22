
import torch 
import math
from torch import optim,nn 
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import time
import matplotlib.pyplot as plt
device = torch.device('cuda:3')
import matplotlib.pyplot as plt 
import random;import time
#importing from files
from utils import scoring_func, RMSELoss,get_data,load_data
from data_processing import process_data
from AdamW import AdamW
torch.manual_seed(5)
torch.cuda.manual_seed(5)
np.random.seed(5)
random.seed(5)
torch.backends.cudnn.deterministic=True


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,batch_size, n_layers):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=True)
    def forward(self, x):
        # input shape {batch_size, seq_length, input_dim} # note input_dim= num of features
        h_init=torch.zeros(self.n_layers, self.batch_size, self.hidden_size, device=device)
        c_init=torch.zeros(self.n_layers, self.batch_size, self.hidden_size, device=device)
        output, (h_init,c_init) = self.lstm(x, (h_init,c_init))
        output= F.dropout(torch.tanh(output),p=0.5, training=self.training)
        return output, h_init,c_init
# Attention Decoder

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, seq_length,batch_size,n_layers):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = output_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.n_layers = n_layers
        
        self.attn = nn.Linear(self.hidden_size + self.output_size, self.seq_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.input_size, self.output_size)
        self.lstm = nn.LSTM(output_size, hidden_size,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, h,c, encoder_outputs):
#         set_trace()
        attn_weights = F.softmax(self.attn(torch.cat((input.squeeze(), h.squeeze()), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),encoder_outputs)
        output = torch.cat((input.squeeze(), attn_applied.squeeze()), 1)
        output = self.attn_combine(output).unsqueeze(1)
        output = F.relu(output)
        output, (h,c) = self.lstm(output.view(self.batch_size,1,self.input_size), (h,c))
        output = self.out(output[:,0,:])
        return output, h,c, attn_weights
    
class regressor(nn.Module):
    
    def __init__(self,hidden):
        super(regressor, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, feat):
        out = self.fc1(feat)
        out = F.dropout(F.relu(out),p=0.2, training=self.training)
        out=  self.fc3(out)
        return out

def train(input_data, target_tensor,seq_length,labels, encoder, attention_decoder, predictor,encoder_optimizer, decoder_optimizer,predictor_optimzer, criterion,denorm_factor):
    teacher_forcing_ratio=0.5
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    predictor_optimzer.zero_grad()
    encoder_out,encoder_h,encoder_C = encoder(input_data)
    #pass enocder hidden to decoder
    decoder_h= encoder_h
    decoder_c= encoder_C
    decoder_input = torch.zeros(batch_size, 1,input_size, device=device)
    
    rec_loss = 0
    pred_loss=0
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    outputs=torch.zeros(batch_size, seq_length,input_size, device=device)
# enoder decoder part
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(seq_length):
            decoder_output,decoder_h,decoder_c,decoder_attention  = attention_decoder(decoder_input, decoder_h,decoder_c, encoder_out)
            outputs[:,di,:]=decoder_output
            rec_loss+=criterion(decoder_output, target_tensor[:,di,:])
            decoder_input = input_data[:,di,:] 
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(seq_length):
            decoder_output,decoder_h,decoder_c,decoder_attention  = attention_decoder(decoder_input, decoder_h,decoder_c, encoder_out)
            outputs[:,di,:]=decoder_output
            decoder_input = decoder_output.unsqueeze(1)
            rec_loss+=criterion(decoder_output, target_tensor[:,di,:])
    enc_dec_features= torch.cat((encoder_h,decoder_h),dim=2)
    yPreds=predictor(enc_dec_features) 
    yPreds = yPreds*denorm_factor
    labels=labels*denorm_factor
    test_score = scoring_func(yPreds.squeeze()-labels)
    rmse=RMSELoss()
    pred_loss= rmse(yPreds.squeeze(),labels)
    loss=pred_loss+rec_loss
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    predictor_optimzer.step()
    return pred_loss.detach().item(),test_score,outputs.detach(),encoder, attention_decoder, predictor

def eval_test(encoder,attention_decoder,predictor,criterion,test_dl,seq_length,denorm_factor):      # set eval state for Dropout and BN layers
    encoder.eval()
    predictor.eval()
    attention_decoder.eval()
    loss = 0
    test_pred=[];test_label=[];valid_losses=[];valid_loss=0
    with torch.no_grad():
        for inputs, labels in test_dl:
                # forward pass
                inputs = (inputs.permute(0,2,1)).float().to(device)
                labels = labels.float().to(device)
             
                encoder_out,encoder_h,encoder_C = encoder(inputs)
                decoder_h= encoder_h
                decoder_c= encoder_C   
                decoder_input = torch.zeros(batch_size, 1,input_size, device=device)
                outputs=torch.zeros(batch_size, seq_length,input_size, device=device)
                for di in range(seq_length):
                    decoder_output,decoder_h,decoder_c,decoder_attention  = attention_decoder(decoder_input, decoder_h,decoder_c, encoder_out)
                    outputs[:,di,:]=decoder_output
                    decoder_input = decoder_output.unsqueeze(1)
                enc_dec_features= torch.cat((encoder_h,decoder_h),dim=2)
                predcitions=predictor(enc_dec_features) 
                predcitions = predcitions*denorm_factor
                # append predictions and labels to calculate the score
                test_pred.append(predcitions.unsqueeze(0))
                test_label.append(labels.unsqueeze(0))
                # calculate the loss
                loss = criterion(predcitions.squeeze(), labels)
                # record validation loss
                valid_losses.append(loss.item())
             #########################################
             #  print training/validation statistics #
             #########################################   
    testing_score  = scoring_func((torch.cat(test_pred, dim=2)).squeeze()-torch.cat(test_label,dim=1).squeeze())
    valid_loss = np.average(valid_losses)
    
    return valid_loss,testing_score

def epoch_train(data,train_dl,test_dl,seq_length,encoder, attention_decoder, predictor,criterion):
    start = time.time()
    total_loss = 0  
    learning_rate=3e-4#
    denorm_factor=125

    encoder_optimizer = AdamW(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = AdamW(attention_decoder.parameters(), lr=learning_rate)
    predictor_optimzer = AdamW(predictor.parameters(), lr=learning_rate)
    
    N_epochs={'FD001':11,'FD003':20}
    rmse_epoch=[];score_epoch=[];test_rmse_epoch=[];test_score_epoch=[];
    for epoch in range (N_epochs[data]):
        RMSE=[];SCORE=[]
        encoder.train()
        attention_decoder.train()
        predictor.train()
        for inputs, labels in train_dl:
            labels=labels.float().to(device)
            inputs=(inputs.permute(0,2,1)).float().to(device)
            rmse,score,outputs,encoder, attention_decoder, predictor= train(inputs,inputs,seq_length, labels,encoder,attention_decoder,predictor, encoder_optimizer, decoder_optimizer,predictor_optimzer, criterion,denorm_factor)  
            RMSE.append(rmse)
            SCORE.append(score)
        rmse_epoch.append(np.average(RMSE))
        score_epoch.append(np.average(SCORE))
        test_rmse, test_score=eval_test(encoder,attention_decoder,predictor,criterion,test_dl,seq_length,denorm_factor)
        test_rmse_epoch.append(test_rmse)
        test_score_epoch.append(test_score)
        print(" Epoch:%d == Training RMSE:%f=== Training Scor:%f" % (epoch,np.average(RMSE),np.sum(SCORE)))
    print("====================== End of Training ======================")
    print(" Performance on Test Set ::Datetset:%s::::::Testing RMSE:%f::::::::Testing Scor:%f" % (data,test_rmse,test_score))
    return rmse_epoch,score_epoch,test_rmse_epoch,test_score_epoch,inputs,outputs,encoder, attention_decoder, predictor

if __name__ == "__main__":
    seq_length=30;batch_size=10
    shuffle_stats=True
    criterion=RMSELoss()
    full_results={'FD003':[]}#,'FD003':[]}
    datasets=['FD003'] # FD003 
    print("Evaluation under shuffle status of::%r" % (shuffle_stats))
    for data in datasets:
      print('Start Training on Datset:%s' %(data))
      train_dl,test_dl,dataloders=load_data(data,seq_length,batch_size,shuffle_stats)
      batch=iter(train_dl)
      x,y=next(batch)
      input_size=x.size(1); 
      hidden_size= 32;
      n_layers=1
      output_size=x.size(1)
      encoder = EncoderLSTM(input_size,hidden_size,batch_size,n_layers).to(device)
      attention_decoder=AttnDecoderLSTM(hidden_size, output_size, seq_length,batch_size,n_layers).to(device)
      predictor=regressor(hidden_size).to(device)
      rmse,score,test_rmse,test_score,inputs,outputs,encoder, attention_decoder, predictor=epoch_train(data,train_dl,test_dl,seq_length,encoder, attention_decoder,predictor,criterion)


      total_results = {'training_rmse':rmse,'training_score':score,'testing_rmse':test_rmse,'testing_score':test_score, 'enocder_model':encoder.state_dict(),
     'attn_decoder_model':attention_decoder.state_dict(),
     'predictor_model':predictor.state_dict(),'inputs':inputs,'reconstructed_inputs':outputs}
      full_results[data].append(total_results)
    hyperparameters={'epoch': 11,'lr': 3e-4,'seq_length':30, 'hidden_size':32, 'teacher_forcing':0.5,'predictor_hidden_size':32,'Enoder_dropout':0.2,'batch_size': 10,}
    torch.save({'parameters':hyperparameters,
                 'full_results':full_results,}, 'Final_FD003_11_10_12PM.pt')

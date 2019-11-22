import torch 
from torch import optim,nn
import numpy as np
import torch.nn.functional as F
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import time
device = torch.device('cuda:1')
import random;
#importing from files
from utils import scoring_func, RMSELoss,load_data
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
        output= F.dropout(torch.tanh(output),p=0.2, training=self.training)
        return output, h_init,c_init
    
# normal decoder 
class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size,batch_size,n_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(output_size, hidden_size,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, x, h_init,c_init):
        out, (h,c) = self.lstm(x, (h_init,c_init))
        output = self.out(out[:,0,:])
        return output, h,c

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
        self.fc1 = nn.Linear(36, 18)
        self.fc2 = nn.Linear(18, 18)
        self.fc3 = nn.Linear(18, 1)

    def forward(self, feat):
        out = self.fc1(feat)
        out = F.dropout(F.relu(out),p=0.2, training=self.training)
        out = self.fc2(out)
        out = F.dropout(F.relu(out),p=0.2, training=self.training)
        out=  self.fc3(out)
        return out

def train(input_data, target_tensor,labels, encoder, attention_decoder, predictor,encoder_optimizer, decoder_optimizer,predictor_optimzer, criterion):
    teacher_forcing_ratio=0.5
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    predictor_optimzer.zero_grad()
    # encoding part
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
    yPreds = yPreds*130
    labels=labels*130
    test_score = scoring_func(yPreds.squeeze()-labels)
    rmse=RMSELoss()
    pred_loss= rmse(yPreds.squeeze(),labels)
    loss=pred_loss+rec_loss
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    predictor_optimzer.step()
    return pred_loss.detach().item(),test_score,outputs.detach(),encoder, attention_decoder, predictor


def eval_test(encoder,attention_decoder,predictor,criterion,test_dl,data_identifier):      # set eval state for Dropout and BN layers
    encoder.eval()
    predictor.eval()
    attention_decoder.eval()
    loss = 0
    seq_length=30;denorm_factor=130
    test_pred=[];test_label=[];valid_losses=[];valid_loss=0
#     set_trace()
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

    if data_identifier== 'FD002':
        testing_score  = scoring_func((torch.cat(test_pred[:-1], dim=2)).squeeze()-torch.cat(test_label[:-1],dim=1).squeeze())
        valid_loss = np.average(valid_losses)
    else:
        testing_score  = scoring_func((torch.cat(test_pred[:-2], dim=2)).squeeze()-torch.cat(test_label[:-2],dim=1).squeeze())
        valid_loss = np.average(valid_losses)
    return valid_loss,testing_score

# traing epoch
def epoch_train(data_identifier,encoder, attention_decoder, predictor,criterion):
    start = time.time()
    # parameters
    learning_rate=3e-4
    num_epochs={'FD002':13,'FD004':16}
    encoder_optimizer = AdamW(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = AdamW(attention_decoder.parameters(), lr=learning_rate)
    predictor_optimzer = AdamW(predictor.parameters(), lr=learning_rate)
    
    train_dl,test_dl,dataloders=load_data(data_identifier,seq_length,batch_size,shuffle_stats)
    total_results={'rmse_epoch':[],'score_epoch':[],'test_rmse_epoch':[],
                  'test_score_epoch':[], 'encoder_epoch':[],
                  'predictor_epoch':[],'decoder_epoch':[],
                  'total_inputs':[],
                  'total_outputs':[] }
    for epoch in range (num_epochs[data_identifier]):
        RMSE=[];SCORE=[]
        encoder.train()
        attention_decoder.train()
        predictor.train()
        for inputs, labels in train_dl:
            labels=labels.float().to(device)
            inputs=(inputs.permute(0,2,1)).float().to(device)
            rmse,score,outputs,encoder, attention_decoder, predictor= train(inputs,inputs, labels,encoder,attention_decoder,predictor, encoder_optimizer, decoder_optimizer,predictor_optimzer, criterion)
            RMSE.append(rmse)
            SCORE.append(score)
        total_results['rmse_epoch'].append(np.average(RMSE))
        total_results['score_epoch'].append(np.sum(SCORE))
        total_results['encoder_epoch'].append(encoder.state_dict())
        total_results['decoder_epoch'].append(attention_decoder.state_dict())
        total_results['predictor_epoch'].append(predictor.state_dict())
        total_results['total_inputs'].append(inputs)
        total_results['total_outputs'].append(outputs)
        epoch_time=time.time()-start
        print(" Epoch:%d == Epoch_time:%f == Training RMSE:%f=== Training Score:%f" % (epoch,epoch_time,np.average(RMSE),np.sum(SCORE)))
        total_time=time.time()-start
    print("============= training completed in: %f=====================" %(total_time))
    test_rmse,test_score=eval_test(encoder,attention_decoder,predictor,criterion, test_dl,data_identifier)
    total_results['test_rmse_epoch'].append(test_rmse)
    total_results['test_score_epoch'].append(test_score)
    print(" Performance on Dataset:%s:::Testing RMSE:%f=== Testing Score:%f" %(data_identifier,test_rmse,test_score))
    return total_results


if __name__ == "__main__":
    seq_length=30;batch_size=10
    shuffle_stats=True
    data_identifiers=['FD002','FD004']
    num_run=1
    full_results_run={'FD002':
                      {'rmse_run':[],'score_run':[],'test_rmse_run':[],
                      'test_score_run':[], 'encoder_run':[],'predictor_run':[],
                      'decoder_run':[], 'total_inputs_run':[],'total_outputs_run':[] },

                      'FD004':
                      {'rmse_run':[],'score_run':[],'test_rmse_run':[],
                      'test_score_run':[], 'encoder_run':[],'predictor_run':[],
                      'decoder_run':[], 'total_inputs_run':[],'total_outputs_run':[] }}
    hyperparameters={'epoch': 16,'lr': 3e-4,'seq_length':30, 'hidden_size':18, 'batch_size': 10,}

    for data in data_identifiers:

    #    for i in range(num_run):
    # dimension paramteres 
        input_size=9; hidden_size= 18;batch_size=10;n_layers=1
        output_size=9;seq_length=30; criterion=RMSELoss()
        # initialization
        encoder = EncoderLSTM(input_size,hidden_size,batch_size,n_layers).to(device)
        attention_decoder=AttnDecoderLSTM(hidden_size, output_size, seq_length,batch_size,n_layers).to(device)
        predictor=regressor(hidden_size).to(device)
        #run the training
        print('Start Training on Datset::%s' %(data))
        total_results =epoch_train(data,encoder,attention_decoder,predictor,criterion)

        full_results_run[data]['rmse_run'].append(total_results['rmse_epoch'])
        full_results_run[data]['score_run'].append(total_results['score_epoch'])
        full_results_run[data]['test_rmse_run'].append(total_results['test_rmse_epoch'])
        full_results_run[data]['test_score_run'].append(total_results['test_score_epoch'])
        full_results_run[data]['encoder_run'].append(total_results['encoder_epoch'])
        full_results_run[data]['encoder_run'].append(total_results['decoder_epoch'])
        full_results_run[data]['predictor_run'].append(total_results['predictor_epoch'])
        full_results_run[data]['total_inputs_run'].append(total_results['total_inputs'])
        full_results_run[data]['total_outputs_run'].append(total_results['total_outputs'])

    torch.save({'parameters':hyperparameters,
                'full_results_10_run':full_results_run,},  'Final_FD002_FD004.pt')

# %%
import torch

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from tqdm import tqdm

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10
#register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# %%
# !wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv

# %%
df2 = pd.read_csv('../data/OxCGRT_USA_latest.csv')
df2.head()

# %%
#df2 = df2['CountryName' == 'United States']
df2 = df2[df2['CountryName'] == 'United States']
df2 = df2[df2['Jurisdiction'] == 'NAT_TOTAL']
print(df2.shape)

# remove some columns V2B_Vaccine age eligibility/availability age floor (general population summary)
df2 = df2.drop(['CountryName', 'CountryCode', 'RegionName', 'RegionCode', 'Jurisdiction', 'Date'], axis=1)
df2 = df2.drop(['V2B_Vaccine age eligibility/availability age floor (general population summary)','V2C_Vaccine age eligibility/availability age floor (at risk summary)','MajorityVaccinated'], axis=1)


df2 = df2.iloc[76:, :]
df2 = df2.reset_index(drop=True)
df2.head()

# %%
df = pd.read_csv('../data/time_series_covid19_confirmed_US.csv')
df.head()

# %%
df = df.iloc[:, 11:]
df.head()

# %%
df.isnull().sum().sum()

# %%
daily_cases = df.sum(axis=0)

daily_cases.index = pd.to_datetime(daily_cases.index)
daily_cases = daily_cases[:-47]
daily_cases.reset_index(drop=True, inplace=True)



#df2.index = pd.to_datetime(df2.index)

daily_cases = pd.concat([daily_cases, df2['GovernmentResponseIndex_Average'], df2['ContainmentHealthIndex_Average'], df2['StringencyIndex_Average'], df2['EconomicSupportIndex']], axis=1)
# daily_cases = pd.concat([daily_cases, df2['ContainmentHealthIndex_Average']], axis=1)
# concat daily cases with zero
#daily_cases = pd.concat([daily_cases, pd.Series(np.zeros(1))], axis=1)
add_index = 4
# replace Nan with 0
daily_cases = daily_cases.fillna(0)
daily_cases_index = pd.date_range(start='2020-01-22', periods=len(daily_cases), freq='D')
daily_cases.index = daily_cases_index

daily_cases.head()

# %%
plt.plot(daily_cases[0])
plt.title("Cumulative daily cases");

# %%
daily_cases[0] = daily_cases[0].diff().fillna(daily_cases[0])

# do a mean to smooth the data
daily_cases[0] = daily_cases[0].rolling(window=7).mean()

# replace Nan with 0
daily_cases = daily_cases.fillna(0)

daily_cases.astype(np.float32)
daily_cases.head()

# %%
plt.plot(daily_cases[0])
plt.title("Daily cases");

# %%
days = daily_cases.shape[0]

print('We have data for ' + str(days))

# %%
test_data_size = 200

train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size-5:]

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

print(train_data.shape)
print(test_data.shape)

# %%
scaler = RobustScaler()
train_data = train_data.values
test_data = test_data.values
scaler = scaler.fit(train_data)

train_data = scaler.transform(train_data)

test_data = scaler.transform(test_data)

# %%
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# %%
seq_length = 14

X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# %%
X_train.shape

# %%
X_train[:2]

# %%
y_train.shape

# %%
y_train[:2]

# %%
train_data[:10]

# %%

class CoronaVirusPredictor(nn.Module):

  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(CoronaVirusPredictor, self).__init__()

    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
    )

    self.linear = nn.Linear(in_features=n_hidden, out_features=1)

  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden).to(self.device),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden).to(self.device)
    )

  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
        sequences.view(len(sequences), self.seq_len, -1),
        self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred

# %%

def train_model(model,train_data_, train_labels_, test_data_ = None, test_labels_ = None):
  loss_fn = torch.nn.MSELoss(reduction='mean')

  optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
  scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.8)
  num_epochs = 40

  train_hist = np.zeros(num_epochs)
  test_hist = np.zeros(num_epochs)
  tbar = tqdm(range(num_epochs))
  for t in tbar:
    model.reset_hidden_state()

    y_pred = model(train_data_)

    loss = loss_fn(y_pred.float(), train_labels_)

    if test_data_ is not None:
      with torch.no_grad():
        y_test_pred = model(test_data_)
        test_loss = loss_fn(y_test_pred.float(), test_labels_)
      test_hist[t] = test_loss.item()

      if t % 10 == 0:
        tbar.set_description(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 10 == 0:
      tbar.set_description(f'Epoch {t} train loss: {loss.item()}')

    train_hist[t] = loss.item()

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
    scheduler.step()

  return model.eval(), train_hist, test_hist

# %%
model = CoronaVirusPredictor(
  n_features=add_index+1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
model = model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)


model, train_hist, test_hist = train_model(
  model, 
  X_train, 
  y_train[:,0].reshape(-1,1), 
  X_test, 
  y_test[:,0].reshape(-1,1)
)

# %% [markdown]
# the train and test loss:

# %%
plt.plot(train_hist, label="Training loss")
plt.plot(test_hist, label="Test loss")
#plt.ylim((0, 55))
plt.legend();

# %%
with torch.no_grad():
  preds = []
  for _ in range(len(X_test)):
    test_seq = X_test[_].view(1, seq_length, add_index+1).float()
    y_test_pred = model(test_seq)
    pred = torch.flatten(y_test_pred).item()
    preds.append(pred)
    
print(preds)

# %%
true_cases = scaler.inverse_transform(y_test.cpu().flatten().numpy().reshape(-1,add_index+1)).flatten()
true_cases = true_cases.reshape(-1,add_index+1)[:,0]
true_cases = true_cases[5:]

preds = np.array(preds).reshape(-1,1)
preds = np.concatenate((preds, np.zeros((preds.shape[0], add_index))), axis=1)
preds[:,1:] = y_test.cpu().numpy()[:,1:]


predicted_cases = scaler.inverse_transform(preds).flatten()
predicted_cases = predicted_cases.reshape(-1,add_index+1)[:,0]
predicted_cases = predicted_cases[5:]


# %%

plt.plot(
  daily_cases.index[:len(train_data)], 
  scaler.inverse_transform(train_data[:].reshape(-1,add_index+1))[:,0].flatten(),
  label='Historical Daily Cases'
)

plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)], 
  true_cases,
  label='Real Daily Cases'
)

plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)], 
  predicted_cases, 
  label='Predicted Daily Cases'
)

plt.legend();

# %%
# get accuracy use %
# print('Accuracy: ', (np.abs(true_cases - predicted_cases)/true_cases).mean())



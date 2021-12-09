from functools import lru_cache
import numpy as np
import pandas as pd
from pandas._libs import missing
import torch
from torch._C import Value
import torch.distributions as td
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
import datetime
import math
import copy
import os
import ujson
import pickle
import regex  as re

def generate_dataset( dset_name, **kwargs ):
    if dset_name == 'toy':
        return generate_dataset_toy(**kwargs)
    
    elif dset_name == "australia_rain":
        return AustraliaRainDataset.get_dataset(**kwargs)
    
    else:
        raise NotImplementedError
    
def generate_dataset_toy( input_shape, sample_size, **kwargs):
    # Load Datasets
    target_func_coeffs = { 'c':0, 'x':torch.randint(0,3, input_shape), 'x^2':torch.randint(0,3,input_shape) }
    tfc = target_func_coeffs
    
    # Sampling Dataset
    loc = torch.zeros(input_shape)
    scale = torch.ones(input_shape)
    ds_train, ds_val, ds_test = ToyDataset.get_dataset(
            
            input_distribution='mv_lognormal',
            inp_sample_params={ 'loc':loc, 'scale':scale },
            
            sample_size=sample_size,

            target_func= lambda inp: torch.sum( tfc['c'] + inp*tfc['x'] + inp*tfc['x^2'], dim=-1 ),

            noise_method = 'random_guassian',
            noise_sample_params= { 'loc':0, 'scale':0.1 }

            )
    
    return ds_train, ds_val, ds_test, None

class ToyDataset(Dataset):

    def __init__(self, features, target ):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.target[idx]
        return feature, target
    
    @staticmethod
    def get_dataset(  input_distribution='uniform',
                                sample_size=1000,
                                inp_sample_params = None,
                                target_func = lambda args: torch.sum(args),
                                noise_method = 'random',
                                noise_sample_params = None,
                                ):

        MAP_DISTR_SAMPLE = {
            'uniform': lambda lb, ub: td.Uniform(lb, ub),
            
            'mv_uniform': lambda lb=0, ub=1 : td.Independent( td.Uniform(lb, ub), 1 ),

            'mv_normal': lambda loc=torch.zeros((6,)), covariance_matrix=torch.eye(6): td.MultivariateNormal( loc , covariance_matrix ),
            
            'mv_lognormal': lambda loc=torch.zeros((6,)), scale=torch.ones( (6,) ) : td.Independent( td.LogNormal( loc , scale ), 1 )

        }


        distr =  MAP_DISTR_SAMPLE[input_distribution](**inp_sample_params)
        X = distr.sample( (sample_size,) )
            #TODO - test situations where more noise is added based on the quantile that Y is in
        X_pertubed = ToyDataset.add_noise(X, noise_method, noise_sample_params) #add noise to X instead of Y
        Y = target_func( X_pertubed )
        if Y.ndim==1:
            Y = Y.unsqueeze(-1)
        # Y = ToyDataset.add_noise(Y, noise_method, noise_sample_params)

        ds = ToyDataset(X,Y)

        train_idx_start = 0
        val_idx_start = int(0.6*sample_size)
        test_idx_start = int(0.8*sample_size)

        # Dividing into train, test, val
        ds_train = Subset(ds, indices = list( range(train_idx_start,val_idx_start) ) ) 
        ds_val = Subset(ds, indices = list( range(val_idx_start, test_idx_start) ))
        ds_test = Subset(ds, indices = list( range(val_idx_start, test_idx_start) ))

        return ds_train, ds_val, ds_test

    @staticmethod
    def add_noise( target,
        method='increasing_at_extremes',
        noise_sample_params=None,
        **kwargs):

        assert method in [ 'random_guassian','increasing_at_extremes', 'increasing_at_maximum', 'increasing_at_minimum' ,'intervals']
        
        if method == 'random_guassian':
            target = target + td.Normal(**noise_sample_params ).sample( tuple(target.shape) ) 
            # target = torch.where( target<0.0, torch.tensor(0.0), target)
            target.clamp_min(0.00)

        # Add noise proportional to decile the data is in
        elif method == 'increasing_at_extremes':
            pass
        # Add relatively more noise to the max deciles
        elif method == 'increasing_at_maximum':
            pass

        # Add relatively more noise to the minimum deciles
        elif method == 'increasing_at_minimum':
            pass

        # Add more noise proportional to the size of the value
        elif method == 'intervals':
            pass
        else:
            raise ValueError
        
        return target

class AustraliaRainDataset(Dataset):
    """
        Dataset source: https://www.kaggle.com/fredericods/forecasting-rain/data?select=weatherAUS.csv

        This dataset contains about 10 years of daily weather observations from numerous Australian weather stations.
        
        This dataset provides point estimates.

        The target RainTomorrow means: Did it rain the next day? Yes or No.

        Note: You should exclude the variable Risk-MM when training your binary classification model. If you don't exclude it, you will leak the answers to your model and reduce its predictability. Read more about it here.

    """
    def __init__(self, features=1, targets=1, lookback=1, location=None) -> None:
        super().__init__()
        self.features = features
        self.targets = targets
        self.lookback = lookback
        self.location = location

        self.create_index_exclude_missing_days()

    def create_index_exclude_missing_days(self ):
        # we do not predict for any day that would have a missing date in it's lookback
        
        self.dates = copy.deepcopy(self.features.index)
        self.indexes_filtrd = list(range(len(self.dates)))

        assert self.features.index.equals(self.targets.index), "Index of features and target are not the same"

        # A list of days missing from the index
        missing_days = pd.date_range(start=self.features.index[0], end=self.features.index[-1]).difference(self.features.index)

        # For each m_day, get the list of days which need m_day for prediction
            # Then remove this list of days from the dates
        for m_day in reversed(missing_days):
            
            li_affected_day = pd.date_range(start = m_day, end = m_day +  pd.DateOffset(days=self.lookback) )
                
                # This range inclusive
            for affected_day in reversed(li_affected_day):
                
                if affected_day in self.dates:

                    index_affected_day = self.dates.get_loc(affected_day)

                    if type(index_affected_day)==slice:
                        raise ValueError
                        
                    elif type(index_affected_day)==int:
                        self.dates = self.dates.drop(affected_day)
                        self.indexes_filtrd.pop(index_affected_day)

                    else:
                        pass


    def __len__(self):
        return len(self.indexes_filtrd) - self.lookback
        
    def __getitem__(self, index):
        ## Note in this formulation index is the day we are predicting for
        index = index + self.lookback

        adj_index = self.indexes_filtrd[index]

        features = self.features.iloc[ adj_index-self.lookback:adj_index ].to_numpy(dtype=np.float32)
        targets = self.targets.iloc[ adj_index-self.lookback:adj_index].to_numpy()
        return features, targets
    
    @staticmethod
    def wind_velocity_calculator(dict_winddirection_radians, wind_direction, wind_speed ):

        assert ( wind_direction in list(dict_winddirection_radians.keys()) ) or np.isnan(wind_direction)
        
        radians = dict_winddirection_radians[wind_direction]
        return wind_speed*math.cos(radians), wind_speed*math.sin(radians)


    @staticmethod
    def get_dataset(start_date = "2008-12-01", end_date="2021-07-03", locations=['Albury'], lookback=6, train_val_test_split = [0.6,0.2,0.2], **kwargs ):

        all_locs = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
        'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
        'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
        'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
        'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
        'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
        'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
        'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
        'AliceSprings', 'Darwin', 'Katherine', 'Uluru']
        assert all( loc in all_locs for loc in locations), "Invalid Location chosen"
        if len(locations) == 0:
            locations = all_locs
        
        # Retreiving Dataset records
        premade_dset_path = os.path.join('Data','australia_rain','premade_dset_record.txt')
        if os.path.exists(premade_dset_path):
            premade_dsets = pd.read_csv( premade_dset_path)
        else:
            premade_dsets = pd.DataFrame( columns=['path','start_date','end_date','locations','lookback','train_val_test_split'] )

        # Query for if existing dataset is made
        query_res = premade_dsets.query( f"start_date == {start_date.replace('-','')} | end_date == {end_date.replace('-','')} | locations == {ujson.dumps(locations)} | lookback == {str(lookback)} | train_val_test_split == {ujson.dumps(train_val_test_split)}" )
        if len(query_res)!=0:
            with open(query_res['path'][0], "rb") as f:
                pkl_dset_dict = pickle.load( f ) #{ 'scaler_features':scaler_features, 'scaler_targets':scaler_targets }
            
            concat_dset_train = pkl_dset_dict['concat_dset_train']
            concat_dset_val  = pkl_dset_dict['concat_dset_val']
            concat_dset_test = pkl_dset_dict['concat_dset_test']
            scaler_features = pkl_dset_dict['scaler_features']
            scaler_targets = pkl_dset_dict['scaler_targets']

        else: # Make dataset from scractch

            data = pd.read_csv("./Data/australia_rain/weatherAUS.csv")

            # Adding Month and Day
            data.insert(loc=1, column='Month', value = data['Date'].apply(lambda x: x[5:7])) #create column "Month"
            data.insert(loc=2, column='Day', value = data['Date'].apply(lambda x: x[7:10])) #create column

            # Selecting specific time subsection
            data.Date = pd.to_datetime(data.Date)
            data = data.loc[ (data.Date >=pd.Timestamp(start_date) ) & ( data.Date <= pd.Timestamp(end_date) ) ]
            
            # Adding Season
            data.insert(loc=3, column='Season', value = data['Month'].replace(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], ['summer','summer', 'summer', 'fall', 'fall', 'fall', 'winter', 'winter', 'winter', 'spring', 'spring', 'spring'])) #create column "Season"
            
            # Converting WindGust into a vector
            # Full Directions -> (N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSw, SW, WSW, W, WWN, NW, NNW, N )
            dict_winddirection_radians = {
                'N':math.pi/2, 'NNE': math.pi/3 , 'NE':math.pi/4, 'ENE':math.pi/6, 'E':0, 'ESE':math.pi*(11/6), 'SE':math.pi*(7/4), "SSE":math.pi*(5/3), 
                'S':math.pi*(3/2), 'SSW':math.pi*(4/3), 'SW':math.pi*(5/4), 'WSW':math.pi*(7/6), 'W':math.pi, 'WNW':math.pi*(5/6), 'NW':math.pi*(3/4), 'NNW':math.pi*(2/3),
                np.nan:0
            }
                # Any element that has E or W in the name - > (NNE, NEE,  )
            data[ ['WindGustU', 'WindGustV'] ] = data.apply( lambda row: AustraliaRainDataset.wind_velocity_calculator( dict_winddirection_radians, row['WindGustDir'], row['WindGustSpeed'] ), axis=1, result_type='expand' )
            data[['WindVelocity9amU', 'WindVelocity9amV']] = data.apply( lambda row: AustraliaRainDataset.wind_velocity_calculator(dict_winddirection_radians, row['WindDir9am'], row['WindSpeed9am'] ), axis=1, result_type='expand' )
            data[['WindVelocity3pmU', 'WindVelocity3pmV']] = data.apply( lambda row: AustraliaRainDataset.wind_velocity_calculator(dict_winddirection_radians, row['WindDir3pm'], row['WindSpeed3pm'] ), axis=1., result_type='expand' )

            
            # Adding RainTomorrowValue
            # data.rename(columns={'RainTomorrow':'RainTomorrowBool'}, inplace=True)
                    
            li_dsets = []
            for loc in locations:
                dataset_loc = data[data.Location == loc ]
                dataset_loc = dataset_loc.sort_values(by='Date')
                # dataset_loc.insert(loc=1, column='Rainfall', value= dataset_loc['Rainfall'].shift(1) )
                # dataset_loc.insert(loc=1, column='RainToday', value= dataset_loc['RainTomorrow'].shift(-1) )

                dataset_loc = dataset_loc.iloc[1:,:]
                li_dsets.append(dataset_loc)
            
            data = pd.concat(li_dsets)
            data = data.reset_index()

            # Drop low quality columns
            # # The variables Sunshine, Evaporation, Cloud3pm, Cloud9am were removed because they had a low fill percentage
            # # Location was removed, since we want forecast rain regardless the location.
            # # Date, Month, Day and were removed, because Season is going to be used instead.
            # # RISK_MM was removed to avoid data leakage.
            # # Rainfall and RainTomorrow are removed to replicate TRUNET settings
            # # WindGustDir, WindGustSpeed, WindDir9am, WindSpeed9am, WindDir3pm, WindSpeed3pm are dropped since they have been replace continous velocity
            data.set_index(['Date'], inplace=True)
            data_final_variables = data.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am', 'Month', 'Day', 'RISK_MM','RainTomorrow',
                                                        'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindSpeed9am', 'WindDir3pm', 'WindSpeed3pm'],axis=1)
            data_final_variables = data_final_variables.dropna()

            targets_raw = data_final_variables[['RainToday','Rainfall','Location']]
            features_raw = data_final_variables.drop(columns = ['RainToday','Rainfall'])
            
            # Scaling Features
            scaler_features = StandardScaler()

            types_aux = pd.DataFrame(features_raw.dtypes)
            types_aux.reset_index(level=0, inplace=True)
            types_aux.columns = ['Variable','Type']
            numerical_feature = list(types_aux[types_aux['Type'] == 'float64']['Variable'].values)

            features_minmax_transform = pd.DataFrame(data = features_raw)
            features_minmax_transform[numerical_feature] = scaler_features.fit_transform(features_raw[numerical_feature])

                # One Hot Encoding
            location_column = features_minmax_transform['Location']
            features_minmax_transform = features_minmax_transform.drop( columns=['Location'],axis=1)
            features_final = pd.get_dummies(features_minmax_transform)
            features_final['Location'] = location_column
            

            # Scaling Targets
            scaler_targets = MinMaxScaler(feature_range=(1,2))

            types_aux = pd.DataFrame(targets_raw.dtypes)
            types_aux.reset_index(level=0, inplace=True)
            types_aux.columns = ['Variable','Type']
            numerical_target = list(types_aux[types_aux['Type'] == 'float64']['Variable'].values)

            target_transform = pd.DataFrame(data = targets_raw )
            target_transform[numerical_target] = scaler_targets.fit_transform(targets_raw[numerical_target])

            target_transform['RainToday'] = target_transform['RainToday'].replace(['Yes', 'No'], [1,0])
            targets_final = target_transform

            # Creating seperate datasets for each location
            li_dsets_train = []
            li_dsets_val   = []
            li_dsets_test  = []


            for loc in locations:
                
                X_loc = features_final[ features_final.Location.str.contains(loc) ]
                Y_loc = targets_final[ targets_final.Location.str.contains(loc) ]

                total_day_count = (  pd.Timestamp(end_date) - pd.Timestamp(start_date) ).days

                missing_days = pd.date_range(start = start_date, end = end_date ).difference(data.index)
                
                if len(missing_days) == 0:
                    train_start_idx = 0
                    train_end_idx = val_start_idx = int( total_day_count * train_val_test_split[0] )
                    val_end_idx = test_start_idx = val_start_idx + int( total_day_count * train_val_test_split[1] )
                
                else:# Handling missing records (days)
                    train_start_idx = 0
                    train_end_idx = val_start_idx = int( len(X_loc) * train_val_test_split[0] )
                    val_end_idx = test_start_idx = val_start_idx + int( len(X_loc)*train_val_test_split[1] )

                X_train = X_loc.iloc[ train_start_idx:train_end_idx ]
                Y_train = Y_loc.iloc[ train_start_idx:train_end_idx ]

                X_val = X_loc.iloc[ val_start_idx:val_end_idx ]
                Y_val = Y_loc.iloc[ val_start_idx:val_end_idx ]

                X_test = X_loc.iloc[ test_start_idx: ]
                Y_test = Y_loc.iloc[ test_start_idx: ]

                X_train = X_train.drop(axis=1, labels=['Location'])
                Y_train = Y_train.drop(axis=1, labels=['Location'])
                X_val = X_val.drop(axis=1, labels=['Location'])
                Y_val = Y_val.drop(axis=1, labels=['Location'])
                X_test = X_test.drop(axis=1, labels=['Location'])
                Y_test = Y_test.drop(axis=1, labels=['Location'])

                dset_train = AustraliaRainDataset(X_train, Y_train, lookback, loc)
                dset_val = AustraliaRainDataset(X_val, Y_val, lookback, loc)
                dset_test = AustraliaRainDataset(X_test, Y_test, lookback, loc)

                li_dsets_train.append(dset_train)
                li_dsets_val.append(dset_val)
                li_dsets_test.append(dset_test)

            concat_dset_train = torch.utils.data.ConcatDataset(li_dsets_train)
            concat_dset_val = torch.utils.data.ConcatDataset(li_dsets_val)
            concat_dset_test = torch.utils.data.ConcatDataset(li_dsets_test)


            # Saving to file for quicker use next time
            try:
                new_dset_number = int( max( [ re.findall("(?<=/)\d+(?=.pkl)",path_) for path_ in premade_dsets['path'].tolist() ] ) ) + 1
            except ValueError:
                new_dset_number = 0

            os.makedirs(os.path.join('Data','australia_rain','premade_dsets'),exist_ok=True)
            path_ = os.path.join('Data','australia_rain','premade_dsets', f'{str(new_dset_number)}.pkl')
            with open(path_,"wb") as f:
                pickle.dump({'concat_dset_train':concat_dset_train, 'concat_dset_val':concat_dset_val, 'concat_dset_test':concat_dset_test, 'scaler_features':scaler_features, 'scaler_targets':scaler_targets }, f  )
            premade_dsets = premade_dsets.append( {'path':path_,
                                    'start_date':start_date,
                                    'end_date':end_date,
                                    'locations':ujson.dumps(locations),
                                    'lookback':str(lookback),
                                    'train_val_test_split':ujson.dumps(train_val_test_split),
                                     } , ignore_index=True)
            premade_dsets.to_csv(premade_dset_path, index=False)

        #TODO: check if any nans included, decide how to handle
        return concat_dset_train, concat_dset_val, concat_dset_test, scaler_features, scaler_targets
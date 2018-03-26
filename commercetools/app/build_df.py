import pandas as pd 
import numpy as np 

def setup_simple(): 
    '''
    Use presaved data of filtered data frame to save time
    '''
    return pd.read_csv('filtered_df.csv')



def filter_df(df = None, invoiceNos = None, stockCodes = None, max_quantity = None,
              min_quantity = None, early_date = None,late_date = None, month = None,
              max_price = None, min_price = None, max_spent = None, min_spent = None,
              countries = None, returns=True, dropnan=True
              ): 
    ''' 
    Function to filter a pandas dataframe. 
    
    Parameters
    ----------
        df :           pandas dataframe - to filter 
        invoiceNos :   list - of invoice numbers to keep. Default is None, which will keep all. 
        stockCodes:    list - of stock codes to keep. Default is None, which will keep all. 
        max_quantity : float - maximum quanitity of the order product. Default is None.
        min_quantity : float - minimum quanitity of the order product. Default is None.
        early_date :   datetime object - earlist date to keep. Default is None. 
        late_date :    datetime object - latest date to keep. Default is None. 
        month :        int - a specific month to look at (Jan = 1, Feb = 2 ... ) Default is None. 
        max_price :    float - maximum price of the product. Default is None. 
        min_price :    float - minimim price of the product. Default is None. 
        max_spent :    float - maximum price of the order. Default is None. 
        min_spent :    float - mimimum price of the order. Default is None.  
        countries :    list of countries to keep. Default is None, which will keep all.
        returns :      bool - if True, keep the returns, else omit them. 
        dropnan :      bool - if True, drop all nan values. Default is True. 
    
    Returns
    -------
        a pandas DataFrame object 
    '''
    
    if invoiceNos is not None: 
        invalid = []
        valid = df.InvoiceNo.unique()
        for c in invoiceNos: 
            if c not in valid: 
                invalid.append(c)
        if len(invalid) > 0: 
            raise ValueError ('{} are not valid invoice numbers. Pick from {}'.format(invalid, valid))
            
        df = df[df['InvoiceNo'].isin(invoiceNos)]
        
    if stockCodes is not None: 
        df = df[df['StockCode'].isin(stockCodes)]
        
    if max_quantity is not None: 
        df = df[df['Quantity'] <= max_quantity]
        
    if min_quantity is not None:
        if min_quantity <= 0:
            raise ValueError('{} not valid price. Pick a positive value'.format(min_quantity))
        
        df = df[df['Quantity'] >= min_quantity]
        
    if early_date is not None: 
        df = df[df['InvoiceDate_'] > early_date]
        
    if late_date is not None: 
        df = df[df['InvoiceDate_'] < late_date]
    
    if month is not None: 
        if month == 0: raise ValueError('0 not a valid month. Pick between 1-12')
        
        df = df[df['month'] == month]
        
    if max_price is not None:
        df = df[df['UnitPrice'] <= max_price]
        
    if min_price is not None:
        if min_price <= 0:
            raise ValueError('{} not valid price. Pick a positive value'.format(min_price))
        
        df = df[df['UnitPrice'] >= min_price]
        
    if max_spent is not None:
        df = df[df['total_spent'] <= max_spent]
    
    if min_spent is not None:
        if min_spent <= 0:
            raise ValueError('{} not valid price. Pick a positive value'.format(min_spent))
        
        df = df[df['total_spent'] >= min_spent]
        
    if countries is not None: 
        invalid = []
        valid = df.Country.unique()
        for c in countries: 
            if c not in valid: 
                invalid.append(c)
        if len(invalid) > 0: 
            raise ValueError ('{} are not valid country names. Pick from {}'.format(invalid, valid))
        
        df = df[df['Country'].isin(countries)]
    
    if returns == False: 
        df = df[df['return'] == 0]
        
    if dropnan == True: 
        df = df.dropna()
        
    return df


def setup_full(): 
    '''the full data preprocessing step '''
    df = pd.read_csv('data.csv')
    df = df.drop(df.columns[0], axis=1)

    # create date features 

    df['InvoiceDate_'] = pd.DatetimeIndex(df.InvoiceDate)

    def year(x): return x.year
    def month(x): return x.month
    def week(x): return x.week
    def day(x): return x.day
    def hour(x): return x.hour
    def minute(x): return x.minute
    def day_of_week(x) : return x.weekday()

    df['year'] = df.InvoiceDate_.apply(year)
    df['month'] = df.InvoiceDate_.apply(month)
    df['week'] = df.InvoiceDate_.apply(week)
    df['day'] = df.InvoiceDate_.apply(day)
    df['hour'] = df.InvoiceDate_.apply(hour)
    # df['minute'] = df.InvoiceDate_.apply(minute)
    df['day_of_week'] = df.InvoiceDate_.apply(day_of_week)

    # get the total amount that was spent in the transaction 
    def total_spent(x): 
        return x['Quantity'] * x['UnitPrice']  
    df['total_spent'] = df.apply(total_spent, axis=1)

    #get whether it was a return or not 
    def returns(x):
        if x.Quantity <= 0 or x.UnitPrice <= 0: 
            return 1
        else: 
            return 0 
    df['return'] = df.apply(returns, axis=1)

    # filter the data and some formatting - get rid of duplicate user/purchase combinations and create counter 
    def combine(x):
        return x.StockCode + str(x.CustomerID)
    filtered_df = filter_df(df, returns=False, dropnan=True)

    filtered_df['temp'] = filtered_df.apply(combine, axis=1)
    filtered_df['ones'] = np.ones_like(filtered_df.year) 
    filtered_df = filtered_df.drop_duplicates('temp')

    return filtered_df
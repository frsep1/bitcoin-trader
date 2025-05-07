import numpy as np

def original(params, data):
    points = data.df['close'].values
    high = np.add(
            params['weights'][0] * data.WMA(points, params['days'][0], data.SMA(params['days'][0])),
            params['weights'][1] * data.WMA(points, params['days'][1], data.LMA(params['days'][1])),
            params['weights'][2] * data.WMA(points, params['days'][2], data.EMA(params['days'][2], params['alphas'][0]))
            ) / sum(params['weights'][:3])
    low = np.add(
            params['weights'][3] * data.WMA(points, params['days'][3], data.SMA(params['days'][3])),
            params['weights'][4] * data.WMA(points, params['days'][4], data.LMA(params['days'][4])),
            params['weights'][5] * data.WMA(points, params['days'][5], data.EMA(params['days'][5], params['alphas'][1]))
            ) / sum(params['weights'][3:])
    return high, low

def MACD(params, data):
    points = data.df['close'].values
    macd = np.subtract(
            data.WMA(points, params['days'][0], data.EMA(params['days'][0], params['alphas'][0])),
            data.WMA(points, params['days'][1], data.EMA(params['days'][1], params['alphas'][1]))
            )
    signal = data.WMA(macd, params['days'][2], data.EMA(params['days'][2], params['alphas'][2]))
    return macd, signal
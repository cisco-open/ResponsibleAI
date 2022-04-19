import dash
import dash_bootstrap_components as dbc
from redis_util import RedisUtils

external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
        
    ]   


print("??????????????????", __name__)
try:
    print("redis = ", redisUtil)
except:
    redisUtil = RedisUtils()
    

app = dash.Dash(external_stylesheets=external_stylesheets,suppress_callback_exceptions=False)

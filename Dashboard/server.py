import dash
import dash_bootstrap_components as dbc
from redis_util import RedisUtils

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
redisUtil = RedisUtils()
from enum import Enum
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd

AUCTION_URL = 'https://www.cricbuzz.com/cricket-series/ipl-2026/auction/teams'

def format_price(price: str):
    price_str = price.split(' (')[0].replace('₹', '')
    if 'crore' in price_str:
        return int(float(price_str.replace('crore', '')) * 100)
    elif 'lakh' in price_str:
        return int(float(price_str.replace('lakh', '')))
    else:
        return int(price_str)

class PlayerType(str, Enum):
    BA = 'Batter'
    AL = 'All Rounder'
    FA = 'Fast Bowler'
    SP = 'Spinner'
    WK = 'Wicket Keeper'
    UBA = 'Uncapped Batter'
    UWK = 'Uncapped Wicket Keeper'
    UFA = 'Uncapped Fast Bowler'
    USP = 'Uncapped Spinner'
    UAL = 'Uncapped All Rounder'

    @classmethod
    def from_set_name(cls, set_name: str):
        return cls[re.split(r'(\d+)', set_name)[0]].value

class Player:
    def __init__(self, name: str, country: str, type: PlayerType, base_price: int, capping: str):
        self.name = name
        self.country = country
        self.type = type
        self.base_price = base_price
        self.capping = capping

class Team:
    def __init__(self, name: str, budget: int):
        self.name = name
        self.squad: list[Player] = []
        self.budget = budget
        self.total_slots = 25
        self.overseas_slots = 8

        self.squad_url = None
        self.auction_url = None

    def load_squad(self):
        
def load_auction_list():
    df = pd.read_csv('dataset.tsv', sep='\t')
    df.columns = ['serial_number', 'set_number', 'set_name', 'player', 'country', 'c_u_a', 'base_price']
    df['type'] = df['set_name'].apply(PlayerType.from_set_name)
    return df

from enum import Enum
import json
import re
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

class AuctionCategory(str, Enum):
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
        return cls[re.split(r'(\d+)', set_name)[0]]

class PlayerTypeTeam(str, Enum):
    BA = 'Batter'
    AL = 'All Rounder'
    FA = 'Bowler'
    SP = 'Bowler'
    WK = 'Wicket Keeper'
    UBA = 'Batter'
    UWK = 'Wicket Keeper'
    UFA = 'Bowler'
    USP = 'Bowler'
    UAL = 'All Rounder'

    @classmethod
    def from_auction_type(cls, auction_type: AuctionCategory):
        return cls[auction_type.name]

class AuctionSet:
    def __init__(self, set_number: int, set_name: str, players: list["Player"]):
        self.set_number = set_number
        self.set_name = set_name
        self.players = players
        self.remaining_players = players.copy()

    def get_random_player(self):
        import random
        if not self.remaining_players:
            return None
        player = random.choice(self.remaining_players)
        self.remaining_players.remove(player)
        return player
    
    def has_remaining_players(self):
        return len(self.remaining_players) > 0 

    def __repr__(self):
        return f"Set {self.set_name} - {len(self.remaining_players)} remaining players - {self.get_random_player()}"

class Player:
    def __init__(self, name: str, isOverseas: bool, type: PlayerTypeTeam, base_price: int, retained_price: int, capping: str = None, type_auction: AuctionCategory = None):
        self.name = name
        self.isOverseas = isOverseas
        self.type = type
        self.base_price = base_price
        self.retained_price = retained_price
        self.capping = capping

    def __repr__(self):
        return f"{self.name} - {self.type} ({self.base_price}L/{self.retained_price}L)"

class Team:
    def __init__(self, name: str, budget: int):
        self.name = name
        self.squad: list[Player] = []
        self.budget = budget
        self.total_spent = 0
        self.total_slots = 25
        self.overseas_slots = 8

        self.squad_url = None
        self.auction_url = None

    def load_squad(self):
        with open(f'squads/{self.name.lower().replace(" ", "-")}.json', 'r') as f:
            squad = json.load(f)
            for player in squad:
                self.budget -= player['playerPrice']
                self.total_spent += player['playerPrice']
                self.total_slots -= 1
                if player['isOverseas']:
                    self.overseas_slots -= 1

                self.squad.append(
                    Player(
                        player['playerName'], 
                        player['isOverseas'], 
                        player['playingType'],
                        None,
                        player['playerPrice'], 
                    )
                )
    
    def can_afford(self, player_amount: int):
        return self.budget >= player_amount

    def add_player(self, player: Player, price: int):
        player.retained_price = price

        self.budget -= price
        self.total_spent += price
        self.total_slots -= 1
        if player.isOverseas:
            self.overseas_slots -= 1

        self.squad.append(player)

    def get_squad_composition(self) -> dict:
        composition = {}
        for player in self.squad:
            composition[player.type] = composition.get(player.type, 0) + 1
        return composition

    def __repr__(self):
        return f"{self.name} - {self.total_spent}L - {self.total_slots} slots - {self.overseas_slots} overseas slots"

def get_teams():
    with open('teams.json', 'r') as f:
        return json.load(f)

def load_auction_list():
    df = pd.read_csv('dataset.tsv', sep='\t')
    df.columns = ['serial_number', 'set_number', 'set_name', 'player', 'country', 'c_u_a', 'base_price']
    df['auction_category'] = df['set_name'].apply(AuctionCategory.from_set_name)
    df['type_team'] = df['auction_category'].apply(PlayerTypeTeam.from_auction_type)

    auction_sets = []

    for set_number in sorted(df['set_number'].unique()):
        set_df = df[df['set_number'] == set_number]
        set_name = set_df['set_name'].iloc[0]

        players = []
        for _, row in set_df.iterrows():
            players.append(Player(
                row['player'],
                row['country'] != 'India',
                row['type_team'],
                row['base_price'],
                None,
                row['c_u_a'],
                row['auction_category']
            ))

        auction_sets.append(AuctionSet(set_number, set_name, players))

    return auction_sets


if __name__ == '__main__':
    auction_sets = load_auction_list()
    for set in auction_sets:
        print(set.get_random_player())
from datetime import datetime
import json
from typing import List, Optional
import pandas as pd

from utils import AuctionCategory, AuctionPhase, PlayerTypeTeam

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
        return f"{self.name} - {self.type} ({self.base_price}L / {self.retained_price}L)"

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
        return f"{self.name} - {self.total_spent}L spent - {self.total_spent / 12500 * 100:.2f}% of budget - {self.total_slots} slots - {self.overseas_slots} overseas slots"


class BidRecord:
    def __init__(self, team: Team, amount: int):
        self.team = team
        self.amount = amount
        self.timestamp = datetime.now().isoformat()

    def __repr__(self) -> str:
        return f"{self.team.name} - Rs. {self.amount} lacs"


class StateManager:
    def __init__(self, teams: List[Team], auction_sets: List[AuctionSet]):
        self.teams = teams
        self.auction_sets = auction_sets
        
        # Current auction state
        self.phase = AuctionPhase.IDLE
        self.current_set_index = 0
        self.current_player: Optional[Player] = None
        self.current_bid = 0
        
        # Bidding state
        self.active_bidders: List[Team] = []
        self.interested_teams: List[Team] = []
        self.opted_out_teams: List[Team] = []
        self.bid_history: List[BidRecord] = []
        
        # Results tracking
        self.sold_players: List[dict] = []
        self.unsold_players: List[Player] = []

    def get_current_set(self) -> Optional[AuctionSet]:
        """Get the current auction set"""
        if self.current_set_index >= len(self.auction_sets):
            return None
        return self.auction_sets[self.current_set_index]
    
    def load_next_player(self) -> Optional[Player]:
        """Load random player from current set"""
        current_set = self.get_current_set()
        
        # Move to next set if current is exhausted
        while current_set and not current_set.has_remaining_players():
            self.current_set_index += 1
            current_set = self.get_current_set()
        
        # Auction complete?
        if not current_set:
            self.phase = AuctionPhase.AUCTION_COMPLETE
            return None
        
        # Get random player
        self.current_player = current_set.get_random_player()
        self.current_bid = self.current_player.base_price
        self.phase = AuctionPhase.PLAYER_ANNOUNCED
        
        # Reset bidding state
        self.active_bidders = []
        self.interested_teams = []
        self.opted_out_teams = []
        self.bid_history = []
        
        return self.current_player

    def set_interested_teams(self, teams: List[Team]):
        """Set which teams want to bid"""
        self.interested_teams = teams.copy()
    
    def set_active_bidders(self, team_a: Team, team_b: Team):
        """Start bidding between two teams"""
        self.active_bidders = [team_a, team_b]
        self.phase = AuctionPhase.BIDDING_ACTIVE
    
    def record_bid(self, team: Team, amount: int):
        """Record a bid in history"""
        self.current_bid = amount
        self.bid_history.append(BidRecord(team, amount))
    
    def mark_team_opted_out(self, team: Team):
        """Team drops out of bidding"""
        if team in self.active_bidders:
            self.active_bidders.remove(team)
        if team not in self.opted_out_teams:
            self.opted_out_teams.append(team)

    def finalize_player_sold(self, winning_team: Team):
        """Player sold to winning team"""
        if not self.current_player:
            raise ValueError("No current player to sell")
        
        winning_team.add_player(self.current_player, self.current_bid)
        
        self.sold_players.append({
            'player': self.current_player,
            'team': winning_team,
            'price': self.current_bid,
            'set': self.get_current_set().set_name
        })
        
        self.phase = AuctionPhase.PLAYER_SOLD
    
    def finalize_player_unsold(self):
        """No one bought the player"""
        if self.current_player:
            self.unsold_players.append(self.current_player)
        self.phase = AuctionPhase.PLAYER_UNSOLD

def get_teams() -> list[str]:
    with open('teams.json', 'r') as f:
        return json.load(f)

def load_auction_list() -> list[AuctionSet]:
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


def initialize_auction():
    teams_list = get_teams()
    teams: list[Team] = []
    for team_name in teams_list:
        team = Team(team_name, 12500)
        team.load_squad()
        teams.append(team)
    
    auction_sets = load_auction_list()
    
    print(f"Loaded {len(teams)} teams")
    print(f"Loaded {len(auction_sets)} auction sets")
    total_players = sum(len(s.players) for s in auction_sets)
    print(f"Total players in auction: {total_players}")
    
    return teams, auction_sets

if __name__ == '__main__':
    teams, auction_sets = initialize_auction()
    state = StateManager(teams, auction_sets)

    player = state.load_next_player()
    print(f"Player: {player.name}, Base: ₹{player.base_price}L")

    # Mark interested teams
    interested = [teams[0], teams[1], teams[2]]
    state.set_interested_teams(interested)
    print(f"Interested: {[t.name for t in state.interested_teams]}")

    # Start bidding
    state.set_active_bidders(teams[0], teams[1])
    print(f"Active bidders: {[t.name for t in state.active_bidders]}")
    print(f"Phase: {state.phase}")

    # Simulate bids
    state.record_bid(teams[0], player.base_price + 50)
    state.record_bid(teams[1], player.base_price + 100)
    state.record_bid(teams[0], player.base_price + 150)
    print(f"Bid history: {[str(b) for b in state.bid_history]}")

    # One team passes
    state.mark_team_opted_out(teams[1])
    print(f"Opted out: {[t.name for t in state.opted_out_teams]}")

    print(teams[0])

    # Finalize
    print(f"\nBefore sale - {teams[0].name} budget: ₹{teams[0].budget}L")
    state.finalize_player_sold(teams[0])
    print(f"After sale - {teams[0].name} budget: ₹{teams[0].budget}L")
    print(f"Phase: {state.phase}")
    print(f"Total sold: {len(state.sold_players)}")

    print(teams[0])
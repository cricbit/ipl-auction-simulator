import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
from re import I
from typing import Dict, List, Optional
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

from utils import AuctionCategory, AuctionPhase, BidDecision, PlayerTypeTeam

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
        return f"{self.name} - {self.type} ({self.base_price} L / {self.retained_price} L)"
    
    def to_dict(self):
        return {
            "name": self.name,
            "isOverseas": self.isOverseas,
            "type": self.type,
            "price": f"{self.base_price or self.retained_price} L"
        }

class Team:
    def __init__(self, name: str, budget: int):
        self.name = name
        self.squad: list[Player] = []
        self.budget = budget
        self.total_spent = 0
        self.max_slots_remaining = 25
        self.max_overseas_slots_remaining = 8

        self.squad_url = None
        self.auction_url = None

    def load_squad(self):
        with open(f'squads/{self.name.lower().replace(" ", "-")}.json', 'r') as f:
            squad = json.load(f)
            for player in squad:
                self.budget -= player['playerPrice']
                self.total_spent += player['playerPrice']
                self.max_slots_remaining -= 1
                if player['isOverseas']:
                    self.max_overseas_slots_remaining -= 1

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
        self.max_slots_remaining -= 1
        if player.isOverseas:
            self.max_overseas_slots_remaining -= 1

        self.squad.append(player)

    def get_squad_composition(self) -> dict:
        composition = {}
        for player in self.squad:
            composition[player.type] = composition.get(player.type, 0) + 1
        return composition

    def __repr__(self):
        return f"{self.name} - {self.total_spent}L spent - {self.total_spent / 12500 * 100:.2f}% of budget - {self.max_slots_remaining} max slots remaining - {self.max_overseas_slots_remaining} max overseas slots remaining"


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

@dataclass
class TeamStrategy:
    team_name: str
    role_priorities: List[str]
    max_budget_per_player: float
    reserve_budget: int
    aggressiveness: str
    critical_gaps: List[str]
    reasoning: str

    def needs_role(self, role: str) -> bool:
        return role in self.role_priorities[:3]
    
    def can_spend(self, amount: int, budget_remaining: int) -> bool:
        if amount > self.max_budget_per_player:
            return False
        if budget_remaining - amount < self.reserve_budget:
            return False
        return True

class LLMClient:
    def __init__(self, model="gpt-5.2"):
        self.model = model
    
    async def call(self, prompt: str) -> str:
        llm = ChatOpenAI(model=self.model, temperature=0, stream_usage=True)
        response = await llm.ainvoke(prompt)
        return response.content
    
class StrategyManager:
    def __init__(self, llm_client: LLMClient, all_teams: List[Team]):
        self.llm_client = llm_client
        self.all_teams = all_teams
        self.strategies: Dict[str, TeamStrategy] = {}

    async def generate_all_strategies(self, teams: List) -> Dict[str, TeamStrategy]:
        print("\n🧠 Generating team strategies...")
        
        for team in teams:
            strategy = await self.generate_strategy(team)
            self.strategies[team.name] = strategy
            print(f"✓ {team.name}: {strategy.aggressiveness}")
        
        return self.strategies
    
    async def generate_strategy(self, team: Team) -> TeamStrategy:
        squad_comp = team.get_squad_composition()
        
        prompt = f"""
            You are the strategist for {team.name} in IPL 2026 auction.
            CURRENT SITUATION:
                - Budget: ₹{team.budget}L
                - Squad: {len(team.squad)} players
                - Minimum Squad Size: 18 players
                - Maximum Slots remaining: {team.max_slots_remaining}
                - Maximum Overseas slots: {team.max_overseas_slots_remaining}
                - Other Teams: {set(self.all_teams) - set([team])} 

            SQUAD:
                {json.dumps([p.to_dict() for p in team.squad], indent=2)}

            SQUAD COMPOSITION:
                {json.dumps(squad_comp, indent=2)}

            AUCTIONED PLAYER CATEGORIES:
                {[e.value for e in AuctionCategory]}

            Analyze and create auction strategy.

            Keep in mind no stats whatsoever are available for any of the current squad or the players to be auctioned, so that's a limitation for now.
            Keep the requirements very simple keeping that in mind. Also, take into account other teams' budget and squad, thinking how they will be competing.

            Respond ONLY with JSON (no markdown):
            {{
                "role_priorities": ["Role1", "Role2", "Role3"],
                "max_budget_per_player": <number>,
                "reserve_budget": <number>,
                "aggressiveness": "conservative|moderate|aggressive",
                "critical_gaps": ["Gap 1", "Gap 2"],
                "reasoning": "Brief explanation"
            }}
        """

        try:
            response = await self.llm_client.call(prompt)
            response = response.replace("```json", "").replace("```", "").strip()
            strategy_data = json.loads(response)
            
            return TeamStrategy(
                team_name=team.name,
                **strategy_data
            )
        except Exception as e:
            print(f"⚠️  Error for {team.name}: {e}")
            return self._default_strategy(team)
    
    def _default_strategy(self, team) -> TeamStrategy:
        return TeamStrategy(
            team_name=team.name,
            role_priorities=["All Rounder", "Batter", "Bowler"],
            max_budget_per_player=int(team.budget * 0.25),
            reserve_budget=int(team.budget * 0.2),
            aggressiveness="moderate",
            critical_gaps=["Balanced squad needed"],
            reasoning="Default strategy"
        )
    
    def get_strategy(self, team_name: str) -> TeamStrategy:
        return self.strategies.get(team_name)


@dataclass
class BiddingDecision:
    action: BidDecision
    bid_amount: int
    confidence: float
    reasoning: str

    def __repr__(self):
        return f"{self.action.value} @ ₹{self.bid_amount}L ({self.reasoning[:50]}...)"


class BiddingManager:
    def __init__(self, llm_client, strategy_manager):
        self.llm_client = llm_client
        self.strategy_manager = strategy_manager
    
    async def evaluate_interest(self, teams: List, player) -> List:
        print(f"\n🤔 Evaluating interest in {player.name}...")
        
        interested = []
        for team in teams:
            if await self._should_enter(team, player):
                interested.append(team)
                print(f"  ✓ {team.name}")
        
        return interested
    
    async def _should_enter(self, team: Team, player: Player) -> bool:
        strategy = self.strategy_manager.get_strategy(team.name)
        
        if not team.can_afford(player.base_price):
            return False
        if team.max_slots_remaining == 0:
            return False
        if player.isOverseas and team.max_overseas_slots_remaining == 0:
            return False
        
        prompt = f"""
            You are {team.name}'s auction agent.
            YOUR STRATEGY:
                - Priorities: {', '.join(strategy.role_priorities)}
                - Budget: ₹{team.budget}L
                - Slots: {team.max_slots_remaining} remaining

            PLAYER:
                - {player.name}
                - Role: {player.type}
                - Base: ₹{player.base_price}L
                - Overseas: {player.isOverseas}

            Should you enter bidding? Respond ONLY with JSON:
            {{
                "should_enter": true|false,
                "reasoning": "brief explanation"
            }}
        """
        
        try:
            response = await self.llm_client.call(prompt)
            response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)
            return data.get("should_enter", False)
        except Exception as e:
            print(f"  ⚠️  {team.name} decision error: {e}")
            return strategy.needs_role(player.type)


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

async def test_interest():
    teams, auction_sets = initialize_auction()
    
    # Setup managers
    llm = LLMClient()
    strategy_mgr = StrategyManager(llm, teams)
    bidding_mgr = BiddingManager(llm, strategy_mgr)
    state = StateManager(teams, auction_sets)
    
    # Generate strategies
    print("Generating strategies...")
    await strategy_mgr.generate_all_strategies(teams)
    
    # Load player
    player = state.load_next_player()
    print(f"\n🏏 Player: {player.name} ({player.type})")
    print(f"   Base: ₹{player.base_price}L")
    
    # Evaluate interest
    interested = await bidding_mgr.evaluate_interest(teams, player)
    
    print(f"\n✓ {len(interested)} teams interested")
    for team in interested:
        strategy = strategy_mgr.get_strategy(team.name)
        print(f"  - {team.name} (wants: {strategy.role_priorities[:2]})")

if __name__ == '__main__':
    asyncio.run(test_interest())
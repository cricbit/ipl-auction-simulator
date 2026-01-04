import re

from enum import Enum

AUCTION_URL = 'https://www.cricbuzz.com/cricket-series/ipl-2026/auction/teams'

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


class AuctionPhase(str, Enum):
    IDLE = "IDLE"
    PLAYER_ANNOUNCED = "PLAYER_ANNOUNCED"
    BIDDING_ACTIVE = "BIDDING_ACTIVE"
    PLAYER_SOLD = "PLAYER_SOLD"
    PLAYER_UNSOLD = "PLAYER_UNSOLD"
    AUCTION_COMPLETE = "AUCTION_COMPLETE"


class BidDecision(str, Enum):
    BID = "BID"
    PASS = "PASS"
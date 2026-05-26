# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# interfaces/gamecontroller.py
#
#   @description:   Utilities to parse GameController data from simulation/real bridge.

from __future__ import annotations

# Constants (compatible with decider historical usage)
MAX_NUM_PLAYERS = 20

# SPL Team Colors
TEAM_BLUE = 0
TEAM_RED = 1
TEAM_YELLOW = 2
TEAM_BLACK = 3
TEAM_WHITE = 4
TEAM_GREEN = 5
TEAM_ORANGE = 6
TEAM_PURPLE = 7
TEAM_BROWN = 8
TEAM_GRAY = 9

TEAM_COLOR_NAMES = {
    TEAM_BLUE: "BLUE",
    TEAM_RED: "RED",
    TEAM_YELLOW: "YELLOW",
    TEAM_BLACK: "BLACK",
    TEAM_WHITE: "WHITE",
    TEAM_GREEN: "GREEN",
    TEAM_ORANGE: "ORANGE",
    TEAM_PURPLE: "PURPLE",
    TEAM_BROWN: "BROWN",
    TEAM_GRAY: "GRAY",
}

# Penalty Types
PENALTY_NONE = 0
PENALTY_SPL_ILLEGAL_BALL_CONTACT = 1
PENALTY_SPL_PLAYER_PUSHING = 2
PENALTY_SPL_ILLEGAL_MOTION_IN_SET = 3
PENALTY_SPL_INACTIVE_PLAYER = 4
PENALTY_SPL_ILLEGAL_POSITION = 5
PENALTY_SPL_LEAVING_THE_FIELD = 6
PENALTY_SPL_REQUEST_FOR_PICKUP = 7
PENALTY_SPL_LOCAL_GAME_STUCK = 8
PENALTY_SPL_ILLEGAL_POSITION_IN_SET = 9
PENALTY_SPL_PLAYER_STANCE = 10
PENALTY_SPL_ILLEGAL_MOTION_IN_STANDBY = 11
PENALTY_SUBSTITUTE = 14
PENALTY_MANUAL = 15

PENALTY_NAMES = {
    PENALTY_NONE: "NONE",
    PENALTY_SPL_ILLEGAL_BALL_CONTACT: "ILLEGAL_BALL_CONTACT",
    PENALTY_SPL_PLAYER_PUSHING: "PLAYER_PUSHING",
    PENALTY_SPL_ILLEGAL_MOTION_IN_SET: "ILLEGAL_MOTION_IN_SET",
    PENALTY_SPL_INACTIVE_PLAYER: "INACTIVE_PLAYER",
    PENALTY_SPL_ILLEGAL_POSITION: "ILLEGAL_POSITION",
    PENALTY_SPL_LEAVING_THE_FIELD: "LEAVING_THE_FIELD",
    PENALTY_SPL_REQUEST_FOR_PICKUP: "REQUEST_FOR_PICKUP",
    PENALTY_SPL_LOCAL_GAME_STUCK: "LOCAL_GAME_STUCK",
    PENALTY_SPL_ILLEGAL_POSITION_IN_SET: "ILLEGAL_POSITION_IN_SET",
    PENALTY_SPL_PLAYER_STANCE: "PLAYER_STANCE",
    PENALTY_SPL_ILLEGAL_MOTION_IN_STANDBY: "ILLEGAL_MOTION_IN_STANDBY",
    PENALTY_SUBSTITUTE: "SUBSTITUTE",
    PENALTY_MANUAL: "MANUAL",
}

# Game States
STATE_INITIAL = 0
STATE_READY = 1
STATE_SET = 2
STATE_PLAYING = 3
STATE_FINISHED = 4
STATE_STANDBY = 5

STATE_NAMES = {
    STATE_INITIAL: "STATE_INITIAL",
    STATE_READY: "STATE_READY",
    STATE_SET: "STATE_SET",
    STATE_PLAYING: "STATE_PLAYING",
    STATE_FINISHED: "STATE_FINISHED",
    STATE_STANDBY: "STATE_STANDBY",
}

STATE_NAME_TO_INT = {v: k for k, v in STATE_NAMES.items()}

SET_PLAY_NAMES = {
    0: "SET_PLAY_NONE",
    1: "SET_PLAY_KICK_OFF",
    2: "SET_PLAY_KICK_IN",
    3: "SET_PLAY_GOAL_KICK",
    4: "SET_PLAY_CORNER_KICK",
    5: "SET_PLAY_DIRECT_FREE_KICK",
    6: "SET_PLAY_INDIRECT_FREE_KICK",
    7: "SET_PLAY_PENALTY_KICK",
}


class GameController:
    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.get_logger()

        config = agent.get_config()
        self.team = int(config.get("team_id", 0))
        # local player id in config is 0-based in this repo
        self.player = int(config.get("id", 0))
        self.color = str(config.get("color", "red")).lower()

        self.game_state = "STATE_INITIAL"
        self.game_state_int = STATE_INITIAL
        self.kicking_team = None
        self.kicking_side = None
        self.secondary_state = None
        self.secondary_state_name = None
        self.set_play = 0
        self.set_play_name = "SET_PLAY_NONE"
        self.placement = None
        self.kick_off = False

        self.penalized_time = 0
        self.penalty = 0
        self.team_color = TEAM_RED if self.color == "red" else TEAM_BLUE
        self.team_color_name = TEAM_COLOR_NAMES.get(self.team_color, "UNKNOWN")
        self.player_penalty_name = "NONE"
        self.can_kick = False
        self.secs_remaining = 0
        self.secondary_time = 0
        self.score = 0
        self.opponent_score = 0

    @staticmethod
    def _safe_int(v, default=0) -> int:
        try:
            return int(v)
        except Exception:
            return int(default)

    def _state_from_packet(self, game_data: dict) -> tuple[int, str]:
        state_raw = game_data.get("state", game_data.get("state_name", STATE_INITIAL))
        if isinstance(state_raw, str):
            if state_raw in STATE_NAME_TO_INT:
                s_int = STATE_NAME_TO_INT[state_raw]
            else:
                # map from simplified play mode schema
                pm = game_data.get("play_mode", "")
                if pm == "GameOver":
                    s_int = STATE_FINISHED
                elif pm in ("BeforeKickOff",):
                    s_int = STATE_READY
                else:
                    s_int = STATE_PLAYING
        else:
            s_int = self._safe_int(state_raw, STATE_INITIAL)
        return s_int, STATE_NAMES.get(s_int, "STATE_INITIAL")

    def _set_play_from_packet(self, game_data: dict) -> tuple[int, str]:
        sp_raw = game_data.get("set_play", 0)
        if isinstance(sp_raw, str):
            name = sp_raw if sp_raw.startswith("SET_PLAY_") else f"SET_PLAY_{sp_raw.upper()}"
            for k, v in SET_PLAY_NAMES.items():
                if v == name:
                    return k, v
            return 0, name
        sp_int = self._safe_int(sp_raw, 0)
        name = game_data.get("set_play_name", SET_PLAY_NAMES.get(sp_int, "SET_PLAY_NONE"))
        return sp_int, name

    def _pick_our_team_data(self, teams: list[dict]) -> tuple[dict | None, dict | None]:
        if not teams:
            return None, None
        our = None
        opp = None
        for t in teams:
            if self._safe_int(t.get("team_number"), -1) == self.team:
                our = t
                break
        if our is None:
            # Fallback by jersey color if team numbers are unavailable.
            want_color = TEAM_RED if self.color == "red" else TEAM_BLUE
            for t in teams:
                if self._safe_int(t.get("field_player_colour"), -999) == want_color:
                    our = t
                    break
        if our is None and teams:
            our = teams[0]

        for t in teams:
            if t is not our:
                opp = t
                break
        return our, opp

    def update(self, game_data: dict):
        if not game_data:
            return

        try:
            self.game_state_int, self.game_state = self._state_from_packet(game_data)
            self.secondary_state = game_data.get("secondary_state", 0)
            self.secondary_state_name = game_data.get("secondary_state_name", None)
            self.set_play, self.set_play_name = self._set_play_from_packet(game_data)

            self.kicking_team = game_data.get("kicking_team", None)
            self.kicking_side = game_data.get("kicking_side", None)
            self.placement = game_data.get("drop_in_team", None)
            self.secs_remaining = self._safe_int(game_data.get("secs_remaining", 0), 0)
            self.secondary_time = self._safe_int(game_data.get("secondary_time", 0), 0)

            teams = game_data.get("teams", [])
            if not isinstance(teams, list):
                teams = []
            our_team_data, opp_team_data = self._pick_our_team_data(teams)

            if our_team_data:
                self.team_color = self._safe_int(our_team_data.get("field_player_colour"), self.team_color)
                self.team_color_name = TEAM_COLOR_NAMES.get(self.team_color, "UNKNOWN")
                self.score = self._safe_int(our_team_data.get("score", 0), 0)

                players = our_team_data.get("players", [])
                if isinstance(players, list):
                    idx = self.player if self.player >= 0 else 0
                    if idx < len(players):
                        p_info = players[idx]
                        self.penalty = self._safe_int(p_info.get("penalty", 0), 0)
                        self.penalized_time = self._safe_int(p_info.get("secs_till_unpenalized", 0), 0)
                        self.player_penalty_name = PENALTY_NAMES.get(self.penalty, "UNKNOWN")
                    else:
                        self.penalty = 0
                        self.penalized_time = 0
                        self.player_penalty_name = "NONE"

            if opp_team_data:
                self.opponent_score = self._safe_int(opp_team_data.get("score", 0), 0)

            # Derive kickoff ownership.
            self.kick_off = bool(self.set_play_name == "SET_PLAY_KICK_OFF" and self.kicking_team == self.team)

            # Stricter GameController-like kick permission:
            # - player not penalized
            # - primary state must be playing
            # - during set play, only kicking team can kick
            if self.penalty != PENALTY_NONE:
                self.can_kick = False
            elif self.game_state_int != STATE_PLAYING:
                self.can_kick = False
            elif self.set_play_name in ("SET_PLAY_NONE", None):
                self.can_kick = True
            else:
                self.can_kick = self.kicking_team == self.team

        except Exception as e:
            self.logger.error(f"Error updating GameController: {e}")

    def debug_print(self):
        pass

# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PlayMode:
    BEFORE_KICK_OFF = "BeforeKickOff"
    KICK_OFF_LEFT = "KickOff_Left"
    KICK_OFF_RIGHT = "KickOff_Right"
    PLAY_ON = "PlayOn"
    THROW_IN_LEFT = "KickIn_Left"
    THROW_IN_RIGHT = "KickIn_Right"
    CORNER_KICK_LEFT = "corner_kick_left"
    CORNER_KICK_RIGHT = "corner_kick_right"
    GOAL_KICK_LEFT = "goal_kick_left"
    GOAL_KICK_RIGHT = "goal_kick_right"
    GOAL_LEFT = "Goal_Left"
    GOAL_RIGHT = "Goal_Right"
    FREE_KICK_LEFT = "free_kick_left"
    FREE_KICK_RIGHT = "free_kick_right"
    GAME_OVER = "GameOver"


@dataclass
class RefereeRules:
    half_time: float = 10.0 * 60.0
    ready_time: float = 5.0
    set_time: float = 5.0
    kick_off_time: float = 15.0
    throw_in_time: float = 15.0
    corner_kick_time: float = 15.0
    goal_kick_time: float = 15.0
    free_kick_time: float = 15.0
    goal_pause_time: float = 3.0
    gc_broadcast_delay_after_goal: float = 15.0
    gc_broadcast_delay_after_playing: float = 15.0


class GCState:
    INITIAL = 0
    READY = 1
    SET = 2
    PLAYING = 3
    FINISHED = 4
    STANDBY = 5


class GCSecondaryState:
    NORMAL = 0
    PENALTY_SHOOT = 1
    OVERTIME = 2
    TIMEOUT = 3


class GCSetPlay:
    NONE = 0
    KICK_OFF = 1
    KICK_IN = 2
    GOAL_KICK = 3
    CORNER_KICK = 4
    DIRECT_FREE_KICK = 5
    INDIRECT_FREE_KICK = 6
    PENALTY_KICK = 7


GC_STATE_NAMES = {
    GCState.INITIAL: "STATE_INITIAL",
    GCState.READY: "STATE_READY",
    GCState.SET: "STATE_SET",
    GCState.PLAYING: "STATE_PLAYING",
    GCState.FINISHED: "STATE_FINISHED",
    GCState.STANDBY: "STATE_STANDBY",
}

GC_SECONDARY_STATE_NAMES = {
    GCSecondaryState.NORMAL: "STATE2_NORMAL",
    GCSecondaryState.PENALTY_SHOOT: "STATE2_PENALTYSHOOT",
    GCSecondaryState.OVERTIME: "STATE2_OVERTIME",
    GCSecondaryState.TIMEOUT: "STATE2_TIMEOUT",
}

GC_SET_PLAY_NAMES = {
    GCSetPlay.NONE: "SET_PLAY_NONE",
    GCSetPlay.KICK_OFF: "SET_PLAY_KICK_OFF",
    GCSetPlay.KICK_IN: "SET_PLAY_KICK_IN",
    GCSetPlay.GOAL_KICK: "SET_PLAY_GOAL_KICK",
    GCSetPlay.CORNER_KICK: "SET_PLAY_CORNER_KICK",
    GCSetPlay.DIRECT_FREE_KICK: "SET_PLAY_DIRECT_FREE_KICK",
    GCSetPlay.INDIRECT_FREE_KICK: "SET_PLAY_INDIRECT_FREE_KICK",
    GCSetPlay.PENALTY_KICK: "SET_PLAY_PENALTY_KICK",
}


class MujocoSoccerReferee:
    def __init__(
        self,
        field_length: float,
        field_width: float,
        goal_width: float,
        goal_height: float,
        goalie_area_depth: float = 1.0,
        goalie_area_extra_width: float = 1.2,
        *,
        rules: RefereeRules | None = None,
        red_count: int = 0,
        blue_count: int = 0,
        left_team_number: int = 12,
        right_team_number: int = 32,
        left_team_name: str = "Home",
        right_team_name: str = "Away",
    ):
        self.field_length = float(field_length)
        self.field_width = float(field_width)
        self.goal_width = float(goal_width)
        self.goal_height = float(goal_height)
        self.goalie_area_depth = float(goalie_area_depth)
        self.goalie_area_width = float(goal_width + 2.0 * goalie_area_extra_width)
        self.rules = rules if rules is not None else RefereeRules()
        self.red_count = int(red_count)
        self.blue_count = int(blue_count)
        self.left_team_number = int(left_team_number)
        self.right_team_number = int(right_team_number)
        self.left_team_name = str(left_team_name)
        self.right_team_name = str(right_team_name)

        self.play_time = 0.0
        self.play_mode = PlayMode.BEFORE_KICK_OFF
        self.play_mode_started_at = 0.0
        self.left_score = 0
        self.right_score = 0
        self.state = GCState.INITIAL
        self.secondary_state = GCSecondaryState.NORMAL
        self.set_play = GCSetPlay.NONE
        self.kicking_side = "left"
        self.kicking_team = self.left_team_number
        self.game_phase = 0
        self.secondary_time = 0
        self._broadcast_frozen_until = 0.0
        self._broadcast_frozen_packet: dict[str, Any] | None = None

        self.agent_na_touch_ball: int | None = None
        self.team_na_score: str | None = None
        self._did_act = False

        self.ball_place_pos: tuple[float, float] | None = None
        self._ball_last_contact: int | None = None
        self._left_players = [{"penalty": 0, "secs_till_unpenalized": 0} for _ in range(max(1, self.red_count))]
        self._right_players = [{"penalty": 0, "secs_till_unpenalized": 0} for _ in range(max(1, self.blue_count))]
        self._last_ball_pos = (0.0, 0.0, 0.0)
        self._auto_state_enabled = True

        self.reset()

    def reset(self):
        self.play_time = 0.0
        self.play_mode = PlayMode.BEFORE_KICK_OFF
        self.play_mode_started_at = 0.0
        self.left_score = 0
        self.right_score = 0
        self.agent_na_touch_ball = None
        self.team_na_score = None
        self._did_act = False
        self.ball_place_pos = None
        self._ball_last_contact = None
        self._last_ball_pos = (0.0, 0.0, 0.0)
        self.secondary_state = GCSecondaryState.NORMAL
        self.secondary_time = 0
        self.set_play = GCSetPlay.NONE
        self._broadcast_frozen_packet = None
        self._broadcast_frozen_until = 0.0
        self._auto_state_enabled = True
        for p in self._left_players:
            p["penalty"] = 0
            p["secs_till_unpenalized"] = 0
        for p in self._right_players:
            p["penalty"] = 0
            p["secs_till_unpenalized"] = 0
        self.kick_off("left")

    def set_auto_state_enabled(self, enabled: bool):
        self._auto_state_enabled = bool(enabled)

    @staticmethod
    def _team_from_rid(rid: int | None) -> str | None:
        if rid is None:
            return None
        return "left" if rid < 7 else "right" # not 3vs3?
        #return "left" if rid < 3 else "right"

    @staticmethod
    def _opponent(team: str) -> str:
        return "right" if team == "left" else "left"

    def _team_number_of_side(self, side: str) -> int:
        return self.left_team_number if side == "left" else self.right_team_number

    def _side_of_team_number(self, team_number: int | None) -> str | None:
        if team_number == self.left_team_number:
            return "left"
        if team_number == self.right_team_number:
            return "right"
        return None

    def _start_gc_broadcast_freeze(self, seconds: float):
        if seconds <= 0.0:
            return
        snapshot = self._game_state_packet(include_legacy=True)
        self._broadcast_frozen_until = self.play_time + seconds
        self._broadcast_frozen_packet = snapshot

    def _set_play_mode(self, play_mode: str):
        self.play_mode = play_mode
        self.play_mode_started_at = self.play_time

    def _set_team_mode(self, team: str, left_mode: str, right_mode: str):
        self._set_play_mode(left_mode if team == "left" else right_mode)

    def kick_off(self, team: str):
        self._did_act = True
        self._set_team_mode(team, PlayMode.KICK_OFF_LEFT, PlayMode.KICK_OFF_RIGHT)
        self.state = GCState.READY
        self.set_play = GCSetPlay.KICK_OFF
        self.kicking_side = team
        self.kicking_team = self._team_number_of_side(team)
        self.agent_na_touch_ball = None
        self.team_na_score = team
        self.ball_place_pos = (0.0, 0.0)

    def play_on(self):
        self._did_act = True
        self._set_play_mode(PlayMode.PLAY_ON)
        self.state = GCState.PLAYING
        self.set_play = GCSetPlay.NONE

    def throw_in(self, team: str, ball_x: float, ball_y: float):
        self._did_act = True
        self._set_team_mode(team, PlayMode.THROW_IN_LEFT, PlayMode.THROW_IN_RIGHT)
        self.state = GCState.PLAYING
        self.set_play = GCSetPlay.KICK_IN
        self.kicking_side = team
        self.kicking_team = self._team_number_of_side(team)
        self.agent_na_touch_ball = None
        self.team_na_score = None
        half_wid = 0.5 * self.field_width
        y = -half_wid if ball_y < 0 else half_wid
        self.ball_place_pos = (float(ball_x), float(y))

    def corner_kick(self, team: str, ball_y: float):
        self._did_act = True
        self._set_team_mode(team, PlayMode.CORNER_KICK_LEFT, PlayMode.CORNER_KICK_RIGHT)
        self.state = GCState.PLAYING
        self.set_play = GCSetPlay.CORNER_KICK
        self.kicking_side = team
        self.kicking_team = self._team_number_of_side(team)
        self.agent_na_touch_ball = None
        self.team_na_score = None
        half_len = 0.5 * self.field_length
        half_wid = 0.5 * self.field_width
        x = half_len if team == "left" else -half_len
        y = -half_wid if ball_y < 0 else half_wid
        self.ball_place_pos = (float(x), float(y))

    def goal_kick(self, team: str):
        self._did_act = True
        self._set_team_mode(team, PlayMode.GOAL_KICK_LEFT, PlayMode.GOAL_KICK_RIGHT)
        self.state = GCState.PLAYING
        self.set_play = GCSetPlay.GOAL_KICK
        self.kicking_side = team
        self.kicking_team = self._team_number_of_side(team)
        self.agent_na_touch_ball = None
        self.team_na_score = None
        half_len = 0.5 * self.field_length
        x = -half_len + 0.5 * self.goalie_area_depth if team == "left" else half_len - 0.5 * self.goalie_area_depth
        self.ball_place_pos = (float(x), 0.0)

    def free_kick(self, team: str, ball_x: float, ball_y: float):
        self._did_act = True
        self._set_team_mode(team, PlayMode.FREE_KICK_LEFT, PlayMode.FREE_KICK_RIGHT)
        self.state = GCState.PLAYING
        self.set_play = GCSetPlay.DIRECT_FREE_KICK
        self.kicking_side = team
        self.kicking_team = self._team_number_of_side(team)
        self.agent_na_touch_ball = None
        self.team_na_score = team
        self.ball_place_pos = (float(ball_x), float(ball_y))

    def goal(self, team: str):
        self._did_act = True
        if team == "left":
            self.left_score += 1
            self._set_play_mode(PlayMode.GOAL_LEFT)
        else:
            self.right_score += 1
            self._set_play_mode(PlayMode.GOAL_RIGHT)
        self.state = GCState.READY
        self.set_play = GCSetPlay.KICK_OFF
        self.kicking_side = self._opponent(team)
        self.kicking_team = self._team_number_of_side(self.kicking_side)
        self.agent_na_touch_ball = None
        self.team_na_score = None
        self._start_gc_broadcast_freeze(self.rules.gc_broadcast_delay_after_goal)

    def game_over(self):
        self._did_act = True
        self._set_play_mode(PlayMode.GAME_OVER)
        self.state = GCState.FINISHED
        self.set_play = GCSetPlay.NONE
        self.agent_na_touch_ball = None
        self.team_na_score = None

    def _play_mode_age(self) -> float:
        return self.play_time - self.play_mode_started_at

    def _is_ball_in_left_goal(self, x: float, y: float, z: float) -> bool:
        return x <= -0.5 * self.field_length and abs(y) <= 0.5 * self.goal_width and z <= self.goal_height

    def _is_ball_in_right_goal(self, x: float, y: float, z: float) -> bool:
        return x >= 0.5 * self.field_length and abs(y) <= 0.5 * self.goal_width and z <= self.goal_height

    def _in_left_goalie_area(self, x: float, y: float) -> bool:
        half_len = 0.5 * self.field_length
        return (-half_len <= x <= -half_len + self.goalie_area_depth) and (abs(y) <= 0.5 * self.goalie_area_width)

    def _in_right_goalie_area(self, x: float, y: float) -> bool:
        half_len = 0.5 * self.field_length
        return (half_len - self.goalie_area_depth <= x <= half_len) and (abs(y) <= 0.5 * self.goalie_area_width)

    def _check_fouls(self, active_contact: int | None):
        if self.team_na_score is not None and active_contact is not None and self._ball_last_contact is not None:
            self.team_na_score = None

        if self.agent_na_touch_ball is not None and active_contact is not None and self._ball_last_contact is not None:
            if self.agent_na_touch_ball == self._ball_last_contact and self.agent_na_touch_ball == active_contact:
                team = self._team_from_rid(active_contact)
                if team is not None:
                    self.free_kick(self._opponent(team), self._last_ball_pos[0], self._last_ball_pos[1])
                    return
            self.agent_na_touch_ball = None

    def _check_timeouts(self):
        if self._did_act:
            return
        if not self._auto_state_enabled:
            return

        # GameController-like primary state transitions.
        if self.state == GCState.READY and self._play_mode_age() > self.rules.ready_time:
            self.state = GCState.SET
            self._set_play_mode(PlayMode.BEFORE_KICK_OFF)
            return
        if self.state == GCState.SET and self._play_mode_age() > self.rules.set_time:
            self.state = GCState.PLAYING
            if self.play_mode == PlayMode.BEFORE_KICK_OFF:
                self._set_team_mode(self.kicking_side, PlayMode.KICK_OFF_LEFT, PlayMode.KICK_OFF_RIGHT)
                self._start_gc_broadcast_freeze(self.rules.gc_broadcast_delay_after_playing)
            return

        pm = self.play_mode
        if pm == PlayMode.PLAY_ON:
            return
        age = self._play_mode_age()
        if pm in (PlayMode.KICK_OFF_LEFT, PlayMode.KICK_OFF_RIGHT) and age > self.rules.kick_off_time:
            self.play_on()
            return
        if pm in (PlayMode.THROW_IN_LEFT, PlayMode.THROW_IN_RIGHT) and age > self.rules.throw_in_time:
            self.play_on()
            return
        if pm in (PlayMode.CORNER_KICK_LEFT, PlayMode.CORNER_KICK_RIGHT) and age > self.rules.corner_kick_time:
            self.play_on()
            return
        if pm in (PlayMode.FREE_KICK_LEFT, PlayMode.FREE_KICK_RIGHT) and age > self.rules.free_kick_time:
            self.play_on()
            return
        if pm == PlayMode.GOAL_KICK_LEFT and age > self.rules.goal_kick_time:
            self.play_on()
            return
        if pm == PlayMode.GOAL_KICK_RIGHT and age > self.rules.goal_kick_time:
            self.play_on()
            return
        if pm == PlayMode.GOAL_LEFT and age > self.rules.goal_pause_time:
            self.kick_off("right")
            return
        if pm == PlayMode.GOAL_RIGHT and age > self.rules.goal_pause_time:
            self.kick_off("left")
            return

    def _check_location_triggers(self, ball_x: float, ball_y: float, ball_z: float):
        if self._did_act:
            return
        pm = self.play_mode
        if pm in (PlayMode.GOAL_LEFT, PlayMode.GOAL_RIGHT):
            return

        if self._is_ball_in_left_goal(ball_x, ball_y, ball_z):
            if pm == PlayMode.GOAL_KICK_LEFT:
                self.play_on()
            else:
                self.goal("right")
            return

        if self._is_ball_in_right_goal(ball_x, ball_y, ball_z):
            if pm == PlayMode.GOAL_KICK_RIGHT:
                self.play_on()
            else:
                self.goal("left")
            return

        half_len = 0.5 * self.field_length
        half_wid = 0.5 * self.field_width
        if abs(ball_x) > half_len or abs(ball_y) > half_wid:
            last_team = self._team_from_rid(self._ball_last_contact)
            if ball_x < -half_len:
                if last_team == "left":
                    self.corner_kick("right", ball_y)
                else:
                    self.goal_kick("left")
            elif ball_x > half_len:
                if last_team == "right":
                    self.corner_kick("left", ball_y)
                else:
                    self.goal_kick("right")
            else:
                if last_team is None:
                    self.throw_in("left", ball_x, ball_y)
                else:
                    self.throw_in(self._opponent(last_team), ball_x, ball_y)
            return

        if pm == PlayMode.GOAL_KICK_LEFT and (not self._in_left_goalie_area(ball_x, ball_y)):
            self.play_on()
            return
        if pm == PlayMode.GOAL_KICK_RIGHT and (not self._in_right_goalie_area(ball_x, ball_y)):
            self.play_on()
            return

    def _check_contact_triggers(self, active_contact: int | None):
        if self._did_act:
            return
        if active_contact is None:
            return
        pm = self.play_mode
        if pm == PlayMode.PLAY_ON:
            return
        if pm in (
            PlayMode.KICK_OFF_LEFT,
            PlayMode.KICK_OFF_RIGHT,
            PlayMode.THROW_IN_LEFT,
            PlayMode.THROW_IN_RIGHT,
            PlayMode.CORNER_KICK_LEFT,
            PlayMode.CORNER_KICK_RIGHT,
            PlayMode.FREE_KICK_LEFT,
            PlayMode.FREE_KICK_RIGHT,
        ):
            team = self._team_from_rid(active_contact)
            if team == "left" and self.red_count > 1:
                self.agent_na_touch_ball = active_contact
            elif team == "right" and self.blue_count > 1:
                self.agent_na_touch_ball = active_contact
            self.play_on()

    def update(self, dt: float, ball_x: float, ball_y: float, ball_z: float, active_contact: int | None):
        self._did_act = False
        self._last_ball_pos = (float(ball_x), float(ball_y), float(ball_z))

        if self.play_mode != PlayMode.GAME_OVER:
            self.play_time += max(0.0, float(dt))

        if self.play_time >= self.rules.half_time and self.play_mode != PlayMode.GAME_OVER:
            self.game_over()
            return

        if active_contact is not None and active_contact != self._ball_last_contact:
            self._ball_last_contact = active_contact

        self._check_fouls(active_contact)
        self._check_timeouts()
        self._check_location_triggers(ball_x, ball_y, ball_z)
        self._check_contact_triggers(active_contact)

    def apply_auto_ref_command(self, command5: list[int] | tuple[int, int, int, int, int]):
        if not isinstance(command5, (list, tuple)) or len(command5) != 5:
            return
        global_cmd, team_cmd, player_cmd, player_number, side = [int(x) for x in command5]
        side_name = "left" if side == 0 else "right"

        if global_cmd == 1 and self.state == GCState.INITIAL:
            self.state = GCState.READY
            self.kick_off(self.kicking_side)
        elif global_cmd == 2 and self.state == GCState.READY:
            self.state = GCState.SET
            self._set_play_mode(PlayMode.BEFORE_KICK_OFF)
        elif global_cmd == 3 and self.state in (GCState.SET, GCState.READY):
            self.state = GCState.PLAYING
            self._set_team_mode(self.kicking_side, PlayMode.KICK_OFF_LEFT, PlayMode.KICK_OFF_RIGHT)
            self._start_gc_broadcast_freeze(self.rules.gc_broadcast_delay_after_playing)
        elif global_cmd == 4 and self.state == GCState.PLAYING:
            self.game_over()
        elif global_cmd == 5 and self.state == GCState.FINISHED:
            self.state = GCState.INITIAL
            self._set_play_mode(PlayMode.BEFORE_KICK_OFF)
        elif global_cmd == 6:
            self.secondary_state = GCSecondaryState.TIMEOUT
        elif global_cmd == 7:
            self.rules.half_time += 60.0

        if team_cmd == 1:
            self.goal_kick(side_name)
        elif team_cmd == 2:
            self.throw_in(side_name, self._last_ball_pos[0], self._last_ball_pos[1])
        elif team_cmd == 3:
            self.corner_kick(side_name, self._last_ball_pos[1])
        elif team_cmd == 4:
            self.set_play = GCSetPlay.PENALTY_KICK
            self.state = GCState.PLAYING
            self.kicking_side = side_name
            self.kicking_team = self._team_number_of_side(side_name)
        elif team_cmd == 5:
            self.set_play = GCSetPlay.DIRECT_FREE_KICK
            self.state = GCState.PLAYING
            self.kicking_side = side_name
            self.kicking_team = self._team_number_of_side(side_name)
        elif team_cmd == 6:
            self.set_play = GCSetPlay.INDIRECT_FREE_KICK
            self.state = GCState.PLAYING
            self.kicking_side = side_name
            self.kicking_team = self._team_number_of_side(side_name)
        elif team_cmd == 7:
            self.kicking_side = side_name
            self.kicking_team = self._team_number_of_side(side_name)
        elif team_cmd == 8:
            self.set_play = GCSetPlay.NONE
        elif team_cmd == 9:
            self.goal(side_name)

        if player_cmd != 0:
            self._apply_player_command(side_name, player_number, player_cmd)

    def _apply_player_command(self, side: str, player_number: int, player_cmd: int):
        arr = self._left_players if side == "left" else self._right_players
        idx = max(0, int(player_number) - 1)
        if idx >= len(arr):
            return
        p = arr[idx]
        # Penalty codes are aligned with current decider constants where possible.
        cmd_to_penalty = {
            1: 14,  # substitute
            2: 2,   # pushing
            3: 7,   # request for pickup
            4: 1,   # ball manipulation / illegal ball contact
            5: 0,   # unpenalize
            6: 15,  # red card (manual)
            7: 15,  # yellow card (manual surrogate)
            8: 15,  # warning (manual surrogate)
        }
        penalty = cmd_to_penalty.get(player_cmd, 0)
        p["penalty"] = int(penalty)
        p["secs_till_unpenalized"] = 0 if penalty == 0 else 10

    def _make_team_packet(
        self,
        team_number: int,
        team_name: str,
        score: int,
        field_player_colour: int,
        goalkeeper_colour: int,
        players: list[dict[str, int]],
    ) -> dict[str, Any]:
        return {
            "team_number": int(team_number),
            "team_name": team_name,
            "score": int(score),
            "field_player_colour": int(field_player_colour),
            "goalkeeper_colour": int(goalkeeper_colour),
            "players": [{"penalty": int(p["penalty"]), "secs_till_unpenalized": int(p["secs_till_unpenalized"])} for p in players],
        }

    def _game_state_packet(self, include_legacy: bool = True) -> dict[str, Any]:
        team_left = self._make_team_packet(
            team_number=self.left_team_number,
            team_name=self.left_team_name,
            score=self.left_score,
            field_player_colour=1,  # red
            goalkeeper_colour=3,    # black
            players=self._left_players,
        )
        team_right = self._make_team_packet(
            team_number=self.right_team_number,
            team_name=self.right_team_name,
            score=self.right_score,
            field_player_colour=0,  # blue
            goalkeeper_colour=4,    # white
            players=self._right_players,
        )
        secs_remaining = max(0, int(round(self.rules.half_time - self.play_time)))
        packet: dict[str, Any] = {
            "state": int(self.state),
            "state_name": GC_STATE_NAMES.get(self.state, "STATE_INITIAL"),
            "secondary_state": int(self.secondary_state),
            "secondary_state_name": GC_SECONDARY_STATE_NAMES.get(self.secondary_state, "STATE2_NORMAL"),
            "set_play": int(self.set_play),
            "set_play_name": GC_SET_PLAY_NAMES.get(self.set_play, "SET_PLAY_NONE"),
            "kicking_team": int(self.kicking_team),
            "kicking_side": self.kicking_side,
            "drop_in_team": int(self.kicking_team),
            "drop_in_time": 0,
            "secs_remaining": secs_remaining,
            "secondary_time": int(self.secondary_time),
            "game_phase": int(self.game_phase),
            "teams": [team_left, team_right],
        }
        if include_legacy:
            packet.update(
                {
                    "play_mode": self.play_mode,
                    "play_time": self.play_time,
                    "score_left": self.left_score,
                    "score_right": self.right_score,
                    "team_na_score": self.team_na_score,
                    "scoring_allowed": "both",
                }
            )
        return packet

    def consume_ball_place(self) -> tuple[float, float] | None:
        out = self.ball_place_pos
        self.ball_place_pos = None
        return out

    def game_state_dict(self) -> dict:
        if self._broadcast_frozen_packet is not None and self.play_time < self._broadcast_frozen_until:
            frozen = dict(self._broadcast_frozen_packet)
            frozen["secs_remaining"] = max(0, int(round(self.rules.half_time - self.play_time)))
            frozen["play_time"] = self.play_time
            return frozen
        self._broadcast_frozen_packet = None
        return self._game_state_packet(include_legacy=True)

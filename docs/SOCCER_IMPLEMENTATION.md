# Soccer implementation decisions — working draft

Companion to `autonomous_soccer_environment_rules_rewards_v1.md`.
The reward configuration is now match-outcome-only, as documented below. This records the current partial
implementation, not a completed or experimentally validated ruleset.

## Kicks and flight

Agreed in the tutor session: player movement resolves first, then all ball
flight advances, including a kick selected in this transition. Strength 1
or 2 limits traversal to one or two trajectory cells in that transition.

Current implementation choices for review:

- Integer-grid interpolation uses the Bresenham helper in `soccer/ball.py`.
  It excludes the origin and includes the destination.
- Player contact uses committed player positions. A player arriving on an
  already in-flight ball's current cell receives it before further flight.
  Otherwise every newly traversed cell is checked in trajectory order.
- The current carrier may kick within the range of its current role. A
  player collecting the ball in this transition cannot also kick it.
- Zero-displacement targets are rejected, retaining possession. Off-pitch
  kicks now stop at the last in-bounds cell on the aimed Bresenham path;
  they do not create a boundary restart. Diagnostics appear in
  `last_kick_results`.
- Reaching an unoccupied destination leaves the ball loose. Kicker identity
  persists until reception so delayed receipt can complete a pass. Receiving
  one's own kick is not a pass. Any receipt clears pending kicker identity.
- Completed passes and opponent possession transfers are exposed as
  per-transition records; they do not yet produce rewards.

Goal detection, match progression, explicit dives, discipline and loose-ball
contests are implemented below. Complete-match validation remains necessary.

## Goal detection — working variant v1.1

The goal-detection batch uses the user's proposed three-cell goal mouth:
left x=0 and right x=15, rows 4, 5 and 6. This differs from baseline v1.0's
single goal cell per team. Configure goal_rows=(5,) to retain that geometry.
The reward configuration is now match-outcome-only, as documented below.

A current defending goalkeeper, including a temporary goalkeeper, blocks
only its occupied goal cell; flight into that cell is received normally.
There is no automatic protection of adjacent goal cells. Explicit flight-contact dives are implemented below; broader simultaneous
contact priority still requires validation.
A field player in a goal cell does not prevent a goal. Goal checking
precedes ordinary reception at each traversed cell and follows dribbling.
A goal stops flight and records exactly one score and goal event. Team IDs
are 0 (left) and 1 (right), regardless of who last touched the ball.

Scores are exposed in observations and per-agent info. The goal_scored_by
record is emitted only on the scoring transition. Episode reset clears
scores and the pending goal.

## Match progression

After a confirmed goal, flight stops immediately. At transition end:

- A third goal naturally terminates the match for every agent.
- Otherwise, reaching max_steps truncates the match; compare scores to
  determine the winner, or draw when equal. Any goal on that transition
  is counted before deciding the outcome.
- A third goal on the last transition takes precedence as natural
  termination, without also reporting truncation.
- If play continues, immediately restore standard formations and original
  roles, clear carrying state, and place a never-possessed loose ball at
  (7, 5) when left conceded or (8, 5) when right conceded.
- Restarts happen within the scoring transition, consume no extra transition,
  and produce no further actions/collection events on that transition.
  Score, time and scoring-transition records are preserved.

The next observation contains the restart state for continuing matches and
final state for completed matches. Observations encode match_result as
0 ongoing, 1 left win, 2 right win or 3 draw. Info includes the result,
match_over and winner (None for an ongoing match or draw). Calling step
after completion requires reset first. Match-outcome rewards are connected as documented below;
red-card suspensions and their restoration will be integrated with discipline.


## Reward configuration — match outcomes only

Explicit user decision supersedes Section 8 of the baseline Markdown:
win +1000, loss -1000, and draw -1000 for each team. Goals, passes,
movement, possession changes and disciplinary actions have no intermediate
reward or penalty. The existing SOCCER_REWARDS_V1 constant name is retained
for compatibility, but it is not the full shaped baseline table.

Each completed match emits exactly one outcome event per team. The shared
EventRewardCalculator stores these in RewardResult.shared_by_team; its
individual entries stay zero. team_total(team) is the canonical aggregate
and is not multiplied by roster size. MultiGrid's per-agent reward mapping
copies that team total to each teammate as the learning signal. For team
reporting use last_team_rewards or info[agent_id]['team_rewards'], rather
than summing the replicated per-agent signals.

## Normal tackles — partial implementation

TACKLE uses target_player_id; the environment supplies the roster size to
SoccerAgent so the action space fits 2v2 or 3v3. Older non-tackle action
dictionaries may omit this unused field for compatibility.

Validation uses starting positions and possession: target an opposing
carrier in one of the eight adjacent cells. A tackle action keeps its
player stationary. A supported attempt succeeds when the seeded uniform
random sample is below 0.5. Failure preserves possession and has no reward
penalty under the match-outcome-only configuration.

Current implementation order for review: movement and dribble collision
transfers resolve before tackle outcomes; tackles resolve before ball
travel. On success, scatter uniformly over on-pitch cells in the 3x3
neighborhood of the target's resolved position. Clear flight/pass provenance,
preserve the previous possessing team, and immediately assign possession
if the chosen cell is occupied. Otherwise the ball remains loose.

last_tackle_results and per-agent info['tackles'] expose target, status,
success (None if no physical attempt resolved), and successful scatter
position. These are transition records; persistent statistics remain pending.

Explicitly unsupported cases currently act as diagnostic no-ops:

- Movement-driven possession changes cancel the stale tackle, as documented below.

These must be resolved before experiments. They are not ordinary failed
tackles. Invalid teammate/distant targets are separately diagnosed and do
not sample tackle randomness. Aggressive tackles and red cards are now
connected as described below; the special loose-ball contest is still unimplemented. Normal off-ball
foul restarts are described below.


## Normal off-ball fouls and restarts

An adjacent off-ball tackle against an opponent is now a foul. Detection
uses pre-transition positions, interrupts the whole joint action before
movement or ball flight, and consumes the transition. The fouled player
is the restart taker. No foul or save reward is added.

Agreed separation rule: other players must be at Manhattan distance >=3
from the taker. Keep already distant, non-conflicting players in place.
Reposition the others in ascending agent-ID order into the nearest legal
unoccupied cell, measuring Manhattan distance from their previous cell;
break ties by x then y. Current goalkeeper movement restrictions apply.
The taker's cell (and designated penalty keeper cell) are reserved first.

A foul outside the offending team's goalkeeper area awards possession to
the fouled player at the foul cell. No movement/kick is executed on the
restart transition. On subsequent transitions, only the taker may kick;
all player movement, tackles and dives remain held until a valid kick
launches. Normal joint play resumes on the following transition. last_foul and last_restart (also copied into info) describe
this transition, then clear on the following one.

User-approved goalkeeper areas are x=0..2 / x=13..15, y=3..7.
Movement, penalty classification, shading and boundary tests now agree.
This explicitly supersedes the original baseline rows 2..8.
Penalty spots remain (2,5)/(13,5), independent of the expanded goal mouth.
The defending goalkeeper is randomly placed at its goal x and y=4 or 6.
Other players are repositioned with the same separation rule. The taker
shoots automatically at the center goal cell on the foul transition:
a regular keeper saves with probability .7, a temporary keeper with .3.
A save gives the keeper possession; otherwise normal goal/match processing
applies. This automatic-save logic does not implement regular-play dives.

Multiple simultaneous fouls are diagnostic no-ops with status
multiple_fouls_unsupported until their restart-priority rule is settled.
A single foul takes priority over any simultaneous ordinary tackle/kick.
Red-card suspensions and disciplinary replacement are implemented below.

## Red-card lifecycle and one-player concession

Explicit session rule change: if a confirmed red card leaves a team with
one active player, award exactly one goal to its opponent. Skip the free
kick/penalty entirely. If that goal ends the match, retain final state;
otherwise the existing goal restart restores all players and original roles.
Thus every red card in 2v2 immediately concedes a goal; in 3v3 this applies
to a second suspension before the next goal. This is a rule choice, not a
claim that a lone goalkeeper cannot score through kicks.

Suspension is distinct from RL termination. Suspended players are excluded
from movement, occupancy, reception, tackling and rendering. Structured
observations expose suspended flags and (-1,-1) positions. Omitted actions
for suspended players are accepted; actions supplied for them are ignored.
The fixed-roster reward API still repeats each team's outcome to its roster.

When two or more teammates remain and a goalkeeper is dismissed, select
the active field player closest to its own central goal. Current concrete
distance/tie-break choice is Manhattan distance, then lowest agent ID.
Relocate that player into the nearest unoccupied legal goalkeeper cell
measured from the defended goal, breaking ties by x then y. The replacement
retains its original field-player identity and receives current goalkeeper
movement/kick restrictions and 30% penalty save probability.

A confirmed card is applied before restart placement. Red-card metadata is
available in last_red_card and info. Normal tackles never set the card flag.
Aggressive-tackle probability rolls now supply red_card=True to this
lifecycle hook as described below. Multiple
simultaneous disciplinary conflicts still require explicit resolution rules.


## Aggressive tackles

AGGRESSIVE_TACKLE is a separate action type using the same target_player_id
field. Normal TACKLE remains 50% success with no card risk. A valid aggressive
carrier challenge draws success at .8 and independently draws red-card
risk at .5. The preparation stage runs before movement; if carded, it
interrupts the joint action and supersedes the physical tackle result.
The sampled success is retained for diagnostics, but success remains None
when no physical outcome resolves. No separate failed-tackle event/reward
is applied on a carded attempt.

Aggressive off-ball tackles always foul and independently roll a .5 card
chance; there is no carrier-success draw in that case. A card connects to
the existing suspension/replacement/restart path, including automatic
concession when one active teammate remains. Multiple carrier tackles now resolve as a group, including independent
card draws, as documented below. Carrier-kick conflicts use the tackle-first rule.
Multiple resulting fouls/cards still require their conflict rule.

A non-carded valid carrier tackle uses the sampled .8 success outcome after
movement, preserving the existing scatter/receipt behavior. Same-transition
possession changes cancel the stale physical tackle as documented below. Diagnostics include tackle
intensity, red_card, sampled_success for aggressive carrier attempts, and
physical success when it resolves. Rewards remain match outcomes only.


## Explicit goalkeeper dives

DIVE uses the existing target field as a displacement (encoded 0..6 minus
3), accepting only the eight adjacent cells. Current role must be goalkeeper,
and the destination must satisfy current goalkeeper movement bounds. The
keeper stays in place until an incoming ball actually traverses its selected
target during that transition. Each selected dive gets at most one attempt.

Check the selected dive before goal detection and ordinary reception on
each flight cell, including intermediate cells of strength-two kicks. Regular
goalkeepers save with probability .7 and temporary goalkeepers with .3 in
regular play, matching their penalty probabilities. This is an explicit user
change from the original baseline, which specified .3 only for temporary
penalty saves. Success
moves the keeper to the target, assigns possession and ends flight. Failure
keeps the keeper still and allows the ball to continue normally. A ball
reaching the keeper's currently occupied cell is still caught automatically,
without a dive or random draw.

Current collision choice for review: a dive cannot enter a cell occupied at
transition start, even if vacated, or one occupied after movement. Multiple
eligible dives at the same cell indicate invalid goalkeeper state and raise
an assertion: opposing legal goalkeeper areas cannot share a dive target. Dives resolve both flight contacts and dribble contacts using the rules below.

last_dive_results and info['dives'] report validation, blocked targets,
triggered outcomes and untriggered selections. No save/dive reward is added.

## Initially loose-ball contests

Movement actions now have contest_intensity: 0 normal, 1 aggressive.
Older action dictionaries and legacy movement inputs default to normal.
This intensity has no effect on ordinary empty-cell collisions or teammate
conflicts. It applies only when exactly two opposing legal moves target
the same initially loose-ball cell, empty of players at transition start.

Draw independent success outcomes (.5 normal, .8 aggressive), and an
independent .5 card outcome for each aggressive participant. Just one
success wins; two successes favor higher intensity; equal intensity uses
a seeded 50/50 tie-break. Two failures leave both players in place.
Possession is assigned after movement commits, through existing collection.

Any card cancels the possession contest before committing any joint movement.
A single card uses the existing disciplinary restart at the contested ball
cell, with the other participant as taker. It is a dedicated contest, not
an off-ball tackle; the foul record is for the disciplinary interruption.
One-player automatic concession still replaces this restart where applicable.

Explicitly agreed double-card edge case: when the simultaneous removals
would leave both teams with one active player, award no goals, restore all
players/original roles/standard formations, and place a never-possessed loose
ball at (7,5) or (8,5), chosen equally with the environment RNG. Preserve
scores and elapsed time. No further actions occur on that transition.

Clarified by the user: when both teams retain two active players, use the
same no-goal neutral restart, but keep all suspended players out. Reset only
active players to their formation slots, placing a current temporary keeper
at its defended goal and preserving that temporary role. Select the loose
ball cell equally from (7,5)/(8,5). Suspensions still end at the next goal.
If both teams are reduced to one, retain the restoration exception above.
If just one team is reduced to one, its automatic concession rule applies;
the other team does not also concede. Multi-player mixed-team destination
contests now use teammate-group exclusion as documented below.

last_loose_ball_contest / info['loose_ball_contest'] record participants,
intensities, sampled successes, card draws and winner. Plural red-card
metadata is exposed in info['red_cards']. Rewards remain match outcomes only.


## Tackle versus kick — tackle first

Agreed ordering: a valid tackle against a carrier resolves before that
carrier's selected kick. Success knocks the ball loose and cancels the
kick for this transition, even if scatter immediately returns possession
to the original carrier. Failure allows the kick to launch and travel in
the same transition. Aggressive card interruption supersedes both physical
tackle and kick through the existing disciplinary restart.

Movement resolution still precedes non-carded physical tackles. A player
who was not the carrier before the transition cannot execute a preselected
kick after receiving a scattered ball. last_kick_results distinguishes
cancelled_by_tackle, interrupted, not_carrier and ordinary launch outcomes.
Multiple simultaneous tackles are implemented below; possession-changing
movement now uses the cancellation rule below.


## Multiple tackles against one carrier

Validate each challenge against the shared starting state and sample each
non-interrupted physical outcome independently (.5 normal, .8 aggressive).
All attempts see the same carrier before any physical outcome is applied.
If any succeeds, scatter the ball exactly once, using the common target's
resolved position, and cancel the carrier's kick even if it recovers the
ball immediately. Successful participants record that same scatter cell.
If all fail, possession remains and the carrier's selected kick may proceed.
Draw/processing order is stable by agent ID, independent of action-map order.

Aggressive card draws precede movement. A disciplinary interruption cancels
all physical outcomes, including otherwise valid normal tackles, and all
kicks. Same-team fouls/cards against the same target share one restart.
Apply every such card before checking remaining roster size: if the offending
team falls to one (or fewer) active players, award one conceded goal instead
of multiple restarts/goals. Otherwise award one free kick or penalty.

Different-target/different-team simultaneous fouls now use the rules below. No intermediate tackle or disciplinary reward
is added; only match outcomes reward teams.


## Simultaneous fouls at different targets or by both teams

Record all foul incidents in last_fouls / info['fouls'], snapshotting positions
before any replacement goalkeeper relocates. Apply every red card first,
then evaluate remaining roster counts using the same helper as loose-ball
contest discipline. A single team at one-or-fewer players concedes one goal;
both teams at one-or-fewer restore everyone through a no-goal neutral reset.

If both teams offended and no low-roster exception applies, cancel all
free-kick/penalty awards and restart neutrally. Preserve suspensions/current
roles and reset only active players; select the loose-ball cell equally
from (7,5)/(8,5). If just one team offended, prioritize its penalty-eligible
fouls. Uniformly select one foul among equal-priority candidates using the
environment RNG, ordered by offender ID for reproducibility. If no penalty
is eligible, select among all its free-kick fouls. The selected fouled player
takes the restart, while every foul/card remains recorded.

All ordinary joint actions are cancelled on the foul transition. Single
fouls require no selection draw. Match-outcome-only rewards are unchanged.


## Three-or-more-player destination contests

For each initially empty destination, group valid moving contenders by
team. A team contributing two or more contenders has every such contender
blocked. Resolve the remaining eligible contenders: none means all stay,
one enters without a random contest, two opponents use the existing 50/50
empty-cell contest or special initially loose-ball challenge. This handles
2-left/1-right, 1-left/2-right and 2-left/2-right symmetrically. The last case
blocks everyone. Blocked teammates stay at their starting positions.

Only eligible participants make challenge/card draws. An aggressive movement
choice alone causes no red-card roll when teammate exclusion prevents an
actual loose-ball challenge. Ordinary occupied-start-cell blocking remains
unchanged. Movement record structure/statuses and match-only rewards remain
unchanged.


## Dribbling versus goalkeeper dives

A keeper's explicit adjacent dive can intercept an opponent who successfully
dribbles into the selected cell. After movement/tackle resolution and ball
synchronization, resolve that dive before goal detection. Use the existing
single save draw (.7 regular, .3 temporary), not an additional probability.

On success, move the keeper into the target, give it possession, and return
the dribbler to its starting position. Update the dribbler's movement record
to blocked_by_dive and the keeper's to dived, including resolved positions.
The starting cell is safe because starting occupancy blocks all other entry.
On failure, preserve both movement results and allow normal goal detection.

For this dribble interaction only, exclude the dribbler from destination
occupancy checks. Other starting/final occupants still block a dive. A
keeper cannot use this interaction against its teammate. A stationary or
already-blocked carrier does not trigger a dribble interception. A successful
tackle takes precedence, so a carrier recovering its scattered ball is not
also treated as completing its original dribble.

The chosen goalkeeper area rows 3..7 now also govern penalties and shading;
the earlier boundary discrepancy is resolved.


## Tackles cancelled by movement-driven possession transfer

Validate adjacency/target possession from the starting state. If movement
resolution transfers the ball before a non-carded tackle physically resolves,
record cancelled_possession_changed and the new carrier ID. Leave success
as None: cancellation is not failure. Do not scatter or alter the new
carrier's possession. A normal cancelled tackle makes no success draw;
an aggressive attempt may retain its already sampled success for diagnostics.

Aggressive card interruption still happens before movement, regardless of
its independently sampled success. Thus the card causes the disciplinary
restart instead of allowing the prospective movement transfer. Rewards
remain exclusively match outcomes.

Opening kickoff, off-pitch kick handling and free-kick restrictions are
finalized in the following section.


## Finalized kickoff, boundary and free-kick rules

A default fresh match starts with a never-possessed loose ball at (7,5) or
(8,5), selected equally by self.np_random. Reusing a reset seed reproduces
that choice. Explicit coordinates and None remain configuration overrides
for controlled scenarios; the demo now uses the default randomized kickoff.

For a legal-range kick aimed outside the pitch, build the original aimed
trajectory and retain its on-pitch prefix. Stop at its last in-bounds cell;
do not clamp the endpoint first, since that changes some diagonal lines.
Contact/interception/dive/goal checks still apply to every traversed cell.
An outward kick already at the boundary has no traversable cells; ordinary
same-cell collection can return it immediately to the kicker.

Free kicks enter a mandatory-kick phase. free_kick_taker is included in
observations (-1 outside the phase). Other players' actions are held, and
the taker cannot dribble instead of taking the kick. Waiting/invalid kicks
retain the phase while time advances. A valid kick clears the phase and
travels immediately; other players remain still for that transition and
resume normal action on the following one. Reset/goal/neutral formations
clear the phase. Penalties keep their existing automatic resolution.


## Goalkeeper invariant audit

check_goalkeeper_invariants() validates stable states: at most one active
current goalkeeper per team, with each inside its own legal 3x5 area.
Suspended original goalkeepers are ignored. This hook is intended for match
validation; it is not imposed during intermediate disciplinary mutations.

Exhaustive checking of all 225 opposing legal keeper-position pairs confirms
that their legal adjacent dive-target sets are disjoint. The former shared
dive-target placeholder is now an assertion identifying corrupted state.
The source audit finds no remaining unsupported/NotImplementedError/TODO
branches in src/soccer; full-match integration validation is still required.

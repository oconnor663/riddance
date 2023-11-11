//! https://stackoverflow.com/questions/28418584/how-to-represent-shared-mutable-state/77399498#77399498

use riddance::{Id, Registry};

type PlayerId = Id<Player>;

struct Player {
    resource_ids: Vec<ResourceId>,
    points: i32,
}

type ResourceId = Id<Resource>;

struct Resource {
    owner_id: PlayerId,
}

struct GameState {
    players: Registry<Player>,
    resources: Registry<Resource>,
}

fn new_player(state: &mut GameState) -> PlayerId {
    state.players.insert(Player {
        points: 0,
        resource_ids: Vec::new(),
    })
}

fn new_resource(state: &mut GameState, owner_id: PlayerId) -> ResourceId {
    let new_id = state.resources.insert(Resource { owner_id });
    state.players[owner_id].resource_ids.push(new_id);
    state.players[owner_id].points += 30;
    new_id
}

fn main() {
    let mut state = GameState {
        players: Registry::new(),
        resources: Registry::new(),
    };
    let player_id = new_player(&mut state);
    let resource_id = new_resource(&mut state, player_id);
    assert_eq!(state.resources[resource_id].owner_id, player_id);
    assert!(state.players[player_id].resource_ids.contains(&resource_id));
}

use riddance::{Id, Registry};

struct Person {
    name: String,
    friends: Vec<PersonId>,
}

type People = Registry<Person>;
type PersonId = Id<Person>;

fn new_person(people: &mut People, name: &str) -> PersonId {
    people.insert(Person {
        name: name.into(),
        friends: Vec::new(),
    })
}

fn add_friend(people: &mut People, this_id: PersonId, other_id: PersonId) {
    if people[other_id].name != people[this_id].name {
        people[this_id].friends.push(other_id);
    }
}

fn main() {
    let mut people = People::new();
    let alice_id = new_person(&mut people, "Alice");
    let bob_id = new_person(&mut people, "Bob");
    add_friend(&mut people, alice_id, bob_id);
    add_friend(&mut people, bob_id, alice_id);
    add_friend(&mut people, alice_id, alice_id); // no-op
}

use riddance::{Id, Registry};

type PersonId = Id<Person>;

struct Person {
    name: String,
    friends: Vec<Id<Person>>,
}

fn new_person(people: &mut Registry<Person>, name: &str) -> PersonId {
    people.insert(Person {
        name: name.into(),
        friends: Vec::new(),
    })
}

fn add_friend(people: &mut Registry<Person>, this_id: PersonId, other_id: PersonId) {
    if people[other_id].name != people[this_id].name {
        people[this_id].friends.push(other_id);
    }
}

fn main() {
    let mut people = Registry::new();
    let alice_id = new_person(&mut people, "Alice");
    let bob_id = new_person(&mut people, "Bob");
    add_friend(&mut people, alice_id, bob_id);
    add_friend(&mut people, bob_id, alice_id);
    add_friend(&mut people, alice_id, alice_id); // no-op
}

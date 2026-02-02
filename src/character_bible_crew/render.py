from __future__ import annotations
from jinja2 import Template
from .schemas import CharacterReedsyProfile

TEMPLATE = Template(
"""# {{ p.basics.name }}

**Role:** {{ p.role or "—" }}  
**Brief:** {{ p.brief or "—" }}

## Differentiation Signature
{% for b in p.differentiation_signature -%}
- {{ b }}
{% endfor %}

## Part 1: The Outer Layer

### The Basics
- Name: {{ p.basics.name }}
- Age: {{ p.basics.age or "—" }}
- Gender: {{ p.gender.identity if p.gender else "—" }} ({{ p.gender.pronouns if p.gender else "—" }})  
- Orientation: {{ p.orientation.romantic if p.orientation else "—" }} / {{ p.orientation.sexual if p.orientation else "—" }}
- Place of birth: {{ p.basics.place_of_birth or "—" }}
- Current location: {{ p.basics.current_location or "—" }}
- Nationality: {{ p.basics.nationality or "—" }}
- Education: {{ p.basics.education or "—" }}
- Occupation: {{ p.basics.occupation or "—" }}
- Income: {{ p.basics.income or "—" }}

### Physical Appearance
- Height: {{ p.physical_appearance.height or "—" }}
- Eye color: {{ p.physical_appearance.eye_color or "—" }}
- Hair color: {{ p.physical_appearance.hair_color or "—" }}
- Build: {{ p.physical_appearance.build or "—" }}
- Distinguishing features: {{ p.physical_appearance.distinguishing_features or "—" }}
- Preferred outfit: {{ p.physical_appearance.preferred_outfit or "—" }}
- Glasses: {{ p.physical_appearance.glasses or "—" }}
- Accessories: {{ p.physical_appearance.accessories or "—" }}
- Grooming: {{ p.physical_appearance.grooming or "—" }}
- Tics & mannerisms: {{ p.physical_appearance.tics_and_mannerisms or "—" }}
- Health: {{ p.physical_appearance.health or "—" }}
- Handwriting: {{ p.physical_appearance.handwriting or "—" }}
- Gait: {{ p.physical_appearance.gait or "—" }}

### Speech & Communication
- Style of speech: {{ p.speech_and_communication.style_of_speech or "—" }}
- Tempo: {{ p.speech_and_communication.tempo or "—" }}
- Accent: {{ p.speech_and_communication.accent or "—" }}
- Pitch: {{ p.speech_and_communication.pitch or "—" }}
- Posture: {{ p.speech_and_communication.posture or "—" }}
- Gesturing: {{ p.speech_and_communication.gesturing or "—" }}
- Eye contact: {{ p.speech_and_communication.eye_contact or "—" }}
- Speech impediments: {{ p.speech_and_communication.speech_impediments or "—" }}
- Speech “tics”: {{ p.speech_and_communication.speech_tics or "—" }}
- Preferred curse word: {{ p.speech_and_communication.preferred_curse_word or "—" }}
- Catchphrases: {{ p.speech_and_communication.catchphrases or "—" }}
- Laughter: {{ p.speech_and_communication.laughter or "—" }}
- Smile: {{ p.speech_and_communication.smile_description or "—" }}
- Emotive/readability: {{ p.speech_and_communication.emotive_readability or "—" }}
- Resting face: {{ p.speech_and_communication.resting_face or "—" }}

## Part 2: The Flesh

### The Past
- Hometown: {{ p.past.hometown or "—" }}
- Childhood type: {{ p.past.childhood_type or "—" }}
- Education: {{ p.past.education_details or "—" }}
- Clubs/organizations: {{ (p.past.school_organizations or []) | join(", ") if p.past.school_organizations else "—" }}
- Yearbook most likely to: {{ p.past.yearbook_most_likely_to or "—" }}
- Jobs / résumé: {{ p.past.jobs_resume or "—" }}
- Dream job as a child: {{ p.past.dream_job_child or "—" }}
- Role models: {{ p.past.role_models or "—" }}
- Greatest regret: {{ p.past.greatest_regret or "—" }}
- Hobbies growing up: {{ p.past.hobbies_growing_up or "—" }}
- Favorite place as a child: {{ p.past.favorite_place_childhood or "—" }}
- Change one thing from the past: {{ p.past.change_one_thing_past or "—" }}
- Turning points: {{ p.past.childhood_turning_points or "—" }}
- Earliest memory: {{ p.past.earliest_memory or "—" }}
- Saddest memory: {{ p.past.saddest_memory or "—" }}
- Happiest memory: {{ p.past.happiest_memory or "—" }}
- Clearest memory: {{ p.past.clearest_memory or "—" }}
- Skeletons in the closet: {{ p.past.skeletons_in_closet or "—" }}
- 3 adjectives as a child: {{ (p.past.three_adjectives_child or []) | join(", ") if p.past.three_adjectives_child else "—" }}
- Advice to younger self: {{ p.past.advice_to_younger_self or "—" }}
- Criminal record: {{ p.past.criminal_record or "—" }}

### Family
- Father: {{ p.family.father.relationship_description if p.family.father else "—" }}
- Mother: {{ p.family.mother.relationship_description if p.family.mother else "—" }}
- Siblings: {{ p.family.siblings or "—" }}
- Children: {{ p.family.children or "—" }}
- Extended family: {{ p.family.extended_family or "—" }}
- Economic status: {{ p.family.economic_status or "—" }}
- See family per year: {{ p.family.how_often_they_see_family or "—" }}

### External Relationships
- Closest friends: {{ p.external_relationships.closest_friends or "—" }}
- Other friends: {{ p.external_relationships.other_significant_friends or "—" }}
- Enemies: {{ p.external_relationships.enemies or "—" }}
- Perceived by strangers: {{ p.external_relationships.perceived_by_strangers or "—" }}
- Perceived by colleagues: {{ p.external_relationships.perceived_by_colleagues or "—" }}
- Perceived by authority: {{ p.external_relationships.perceived_by_authority or "—" }}
- Social media platforms: {{ (p.external_relationships.social_media_platforms or []) | join(", ") if p.external_relationships.social_media_platforms else "—" }}
- Social media usage: {{ p.external_relationships.social_media_usage or "—" }}
- Dating profile: {{ p.external_relationships.online_dating_profile or "—" }}
- Group role: {{ p.external_relationships.group_role or "—" }}
- Email response speed: {{ p.external_relationships.email_response_speed or "—" }}
- Wants from relationship: {{ p.external_relationships.wants_from_relationship or "—" }}
- Ideal partner: {{ p.external_relationships.ideal_partner or "—" }}
- Significant other: {{ p.external_relationships.significant_other or "—" }}
- Funeral attendance: {{ p.external_relationships.funeral_attendance or "—" }}

## Part 3: The Core

### Psychology
- Rainy days: {{ p.psychology.rainy_day_behavior or "—" }}
- Street-smart / book-smart: {{ p.psychology.street_smart_vs_book_smart or "—" }}
- Optimist / pessimist: {{ p.psychology.optimist_vs_pessimist or "—" }}
- Introvert / extrovert: {{ p.psychology.introvert_vs_extrovert or "—" }}
- Favorite sound: {{ p.psychology.favorite_sound or "—" }}
- Favorite place in the world: {{ p.psychology.favorite_place_world or "—" }}
- Secrets they keep: {{ p.psychology.secrets_they_keep or "—" }}
- Want most: {{ p.psychology.want_the_most or "—" }}
- Biggest flaw: {{ p.psychology.biggest_flaw or "—" }}
- Biggest strength: {{ p.psychology.biggest_strength or "—" }}
- Biggest fear: {{ p.psychology.biggest_fear or "—" }}
- Biggest accomplishment: {{ p.psychology.biggest_accomplishment or "—" }}
- Perfect happiness: {{ p.psychology.perfect_happiness or "—" }}
- Remembered for: {{ p.psychology.remembered_for or "—" }}
- Favorite quote: {{ p.psychology.favorite_quote or "—" }}
- Moral compass & compromise: {{ p.psychology.moral_compass_and_compromise or "—" }}
- Pet peeves: {{ p.psychology.pet_peeves or "—" }}
- Tombstone: {{ p.psychology.tombstone or "—" }}

### The Present & Future
- Story goal: {{ p.present_and_future.story_goal or "—" }}
- Story motivation: {{ p.present_and_future.story_motivation or "—" }}

## Devil’s Advocate Notes
{% for b in p.clone_risks_and_fixes -%}
- {{ b }}
{% endfor %}
{% if p.notes and p.notes.stress_test_snippets is defined and p.notes.stress_test_snippets %}
## Stress-Test Snippets
*Proof-of-life: how this character sounds under pressure.*
{% for item in p.notes.stress_test_snippets %}
### Scenario
{{ item.scenario }}

```
{{ item.snippet }}
```
{% endfor %}
{% endif %}
{% if p.notes and p.notes.dropped_fields is defined and p.notes.dropped_fields %}
## Provenance: Dropped Fields
*These sections failed schema validation and were set aside rather than blocking the character.*
{% for key, value in p.notes.dropped_fields.items() %}
- **{{ key }}** — removed; raw value preserved in `character.json` under `notes.dropped_fields`
{% endfor %}
{% endif %}

"""
)

def render_markdown(profile: CharacterReedsyProfile) -> str:
    return TEMPLATE.render(p=profile)

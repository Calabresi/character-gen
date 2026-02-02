from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class ReedsyBasics(BaseModel):
    name: str
    age: Optional[str] = None
    place_of_birth: Optional[str] = None
    current_location: Optional[str] = None
    nationality: Optional[str] = None
    education: Optional[str] = None
    occupation: Optional[str] = None
    income: Optional[str] = None

class GenderIdentity(BaseModel):
    identity: str = Field(..., description="Character gender identity (e.g. woman, man, nonbinary)")
    pronouns: str = Field(..., description="Pronouns used in narration/dialogue (e.g. she/her)")

class Orientation(BaseModel):
    romantic: str = Field(..., description="Romantic orientation (e.g. heteroromantic, biromantic, aromantic, unknown)")
    sexual: str = Field(..., description="Sexual orientation (e.g. heterosexual, bisexual, asexual, demisexual, unknown)")
    notes: Optional[str] = Field(None, description="Short clarification; can include 'do not invent' if undecided.")

class PhysicalAppearance(BaseModel):
    height: Optional[str] = None
    eye_color: Optional[str] = None
    hair_color: Optional[str] = None
    build: Optional[str] = None
    distinguishing_features: Optional[str] = None
    preferred_outfit: Optional[str] = None
    glasses: Optional[str] = None
    accessories: Optional[str] = None
    grooming: Optional[str] = None
    tics_and_mannerisms: Optional[str] = None
    health: Optional[str] = None
    handwriting: Optional[str] = None
    gait: Optional[str] = None

class SpeechCommunication(BaseModel):
    style_of_speech: Optional[str] = None
    tempo: Optional[str] = None
    accent: Optional[str] = None
    pitch: Optional[str] = None
    posture: Optional[str] = None
    gesturing: Optional[str] = None
    eye_contact: Optional[str] = None
    speech_impediments: Optional[str] = None
    speech_tics: Optional[str] = None
    preferred_curse_word: Optional[str] = None
    catchphrases: Optional[str] = None
    laughter: Optional[str] = None
    smile_description: Optional[str] = None
    emotive_readability: Optional[str] = None
    resting_face: Optional[str] = None

class PastEducationJobs(BaseModel):
    hometown: Optional[str] = None
    childhood_type: Optional[str] = None
    education_details: Optional[str] = None
    school_organizations: Optional[List[str]] = None
    yearbook_most_likely_to: Optional[str] = None
    jobs_resume: Optional[str] = None
    dream_job_child: Optional[str] = None
    role_models: Optional[str] = None
    greatest_regret: Optional[str] = None
    hobbies_growing_up: Optional[str] = None
    favorite_place_childhood: Optional[str] = None
    change_one_thing_past: Optional[str] = None
    childhood_turning_points: Optional[str] = None
    earliest_memory: Optional[str] = None
    saddest_memory: Optional[str] = None
    happiest_memory: Optional[str] = None
    clearest_memory: Optional[str] = None
    skeletons_in_closet: Optional[str] = None
    three_adjectives_child: Optional[List[str]] = None
    advice_to_younger_self: Optional[str] = None
    criminal_record: Optional[str] = None

class FamilyMember(BaseModel):
    age_if_living: Optional[str] = None
    occupation: Optional[str] = None
    relationship_description: Optional[str] = None

class FamilySection(BaseModel):
    father: Optional[FamilyMember] = None
    mother: Optional[FamilyMember] = None
    siblings: Optional[str] = None
    children: Optional[str] = None
    extended_family: Optional[str] = None
    economic_status: Optional[str] = None
    how_often_they_see_family: Optional[str] = None

class ExternalRelationships(BaseModel):
    closest_friends: Optional[str] = None
    other_significant_friends: Optional[str] = None
    enemies: Optional[str] = None
    perceived_by_strangers: Optional[str] = None
    perceived_by_acquaintances_work: Optional[str] = None
    perceived_by_colleagues: Optional[str] = None
    perceived_by_authority: Optional[str] = None
    perceived_by_friend_circles: Optional[str] = None
    perceived_by_children: Optional[str] = None
    perceived_by_opposite_sex: Optional[str] = None
    perceived_by_extended_family: Optional[str] = None
    social_media_platforms: Optional[List[str]] = None
    social_media_usage: Optional[str] = None
    online_dating_profile: Optional[str] = None
    group_role: Optional[str] = None
    email_response_speed: Optional[str] = None
    depends_on_practical_advice: Optional[str] = None
    depends_on_mentoring: Optional[str] = None
    depends_on_wingman: Optional[str] = None
    depends_on_emotional_support: Optional[str] = None
    depends_on_moral_support: Optional[str] = None
    wants_from_relationship: Optional[str] = None
    ideal_partner: Optional[str] = None
    significant_other: Optional[str] = None
    funeral_attendance: Optional[str] = None

class PsychologyCore(BaseModel):
    rainy_day_behavior: Optional[str] = None
    street_smart_vs_book_smart: Optional[str] = None
    optimist_vs_pessimist: Optional[str] = None
    introvert_vs_extrovert: Optional[str] = None
    favorite_sound: Optional[str] = None
    favorite_place_world: Optional[str] = None
    secrets_they_keep: Optional[str] = None
    want_the_most: Optional[str] = None
    biggest_flaw: Optional[str] = None
    biggest_strength: Optional[str] = None
    biggest_fear: Optional[str] = None
    biggest_accomplishment: Optional[str] = None
    perfect_happiness: Optional[str] = None
    remembered_for: Optional[str] = None
    favorite_quote: Optional[str] = None
    approach_to_power: Optional[str] = None
    approach_to_ambition: Optional[str] = None
    approach_to_love: Optional[str] = None
    approach_to_change: Optional[str] = None
    burning_home_object: Optional[str] = None
    what_bores_them: Optional[str] = None
    what_makes_angry: Optional[str] = None
    what_they_look_for_in_person: Optional[str] = None
    moral_compass_and_compromise: Optional[str] = None
    last_10_books: Optional[List[str]] = None
    fictional_world_to_visit: Optional[str] = None
    if_no_sleep: Optional[str] = None
    pet_peeves: Optional[str] = None
    if_won_lottery: Optional[str] = None
    bucket_list_15: Optional[str] = None
    bucket_list_20: Optional[str] = None
    bucket_list_30: Optional[str] = None
    bucket_list_40: Optional[str] = None
    spotify_top_10_songs: Optional[List[str]] = None
    best_compliment: Optional[str] = None
    elevator_button_twice: Optional[str] = None
    tombstone: Optional[str] = None

class PresentFuture(BaseModel):
    story_goal: Optional[str] = None
    story_motivation: Optional[str] = None

class CharacterReedsyProfile(BaseModel):
    # Canon/identity helpers
    slug: str = Field(..., description="filesystem-safe id, e.g. marin-vale")
    role: Optional[str] = None
    brief: Optional[str] = None

    # Reedsy sections
    basics: ReedsyBasics
    gender: Optional[GenderIdentity] = None
    orientation: Optional[Orientation] = None
    physical_appearance: PhysicalAppearance = Field(default_factory=PhysicalAppearance)
    speech_and_communication: SpeechCommunication = Field(default_factory=SpeechCommunication)
    voice_constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cast-level voice matrix for this character. Keys like: do, avoid, banned_phrases, banned_metaphor_domains, syntax_rules, social_mask."
    )
    past: PastEducationJobs = Field(default_factory=PastEducationJobs)
    family: FamilySection = Field(default_factory=FamilySection)
    external_relationships: ExternalRelationships = Field(default_factory=ExternalRelationships)
    psychology: PsychologyCore = Field(default_factory=PsychologyCore)
    present_and_future: PresentFuture = Field(default_factory=PresentFuture)

    # Anti-clone scaffolding
    differentiation_signature: List[str] = Field(
        default_factory=list,
        description="3–7 bullets that make this character unmistakably NOT the author."
    )
    clone_risks_and_fixes: List[str] = Field(
        default_factory=list,
        description="Devil’s advocate notes: where they feel generic / too similar, and what to change."
    )

    # For debugging / provenance
    notes: Optional[Dict[str, Any]] = None

fb_preset_naming= {
        'adaptive/blurbg-strong': 'adaptive/blurbg_strong', 
        'adaptive/blurbg-subtle': 'adaptive/blurbg_subtle',
        'adaptive/enhance-portait': 'adaptive/enhance_portrait', 
        'adaptive/sky-bluedrama': 'adaptive/sky_bluedrama', 
        'adaptive/subject-pop': 'adaptive/subject_pop', 
        'portraits/lightskin-PL10': 'portraits/light-skin-PL10', 
        'portraits/mediumskin-PM11': 'portraits/medium-skin-PM11', 
        'seasons/autumn-TM01': 'seasons/autumn_TM01', 
        'seasons/spring-SP01': 'seasons/spring_SP01', 
        'seasons/summer-SM01': 'seasons/summer_SM01', 
        'seasons/winter-WN01': 'seasons/winter_WN01', 
        'style/bw-BW01': 'style/bw_BW01', 
        'style/cinematic-CN01': 'style/cinematic_CN01', 
        'style/cinematic2-CN11': 'style/cinematic2_CN11', 
        'style/film-inspired-boldbw': 'style/film_inspired_boldbw', 
        'style/film-inspired-coolbw': 'style/film_inspired_coolbw', 
        'style/film-inspired-warmgold': 'style/film_inspired_warmgold', 
        'style/futuristic-FT01': 'style/futuristic_FT01', 
        'style/vintage-VN01': 'style/vintage_VN01', 
        'subject/landscape-LN01': 'subject/landscape_LN01', 
        'subject/travel-TR01': 'subject/travel_TR01', 
        'subject/travel2-TR11': 'subject/travel2_TR11', 
        'subject/urbanarchitecture-UA08': 'subject/urban-architecture-UA08'}
    
tl_preset_naming = {
    'adaptive/blurbg-strong': 'adaptive/blurbg_strong', 
    'adaptive/blurbg-subtle': 'adaptive/blurbg_subtle', 
    'adaptive/enhance-portait': 'adaptive/enhance_portrait', 
    'adaptive/sky-bluedrama': 'adaptive/sky_bluedrama', 
    'adaptive/subject-pop': 'adaptive/subject_pop', 
    'portraits/lightskin-PL10': 'portraits/light-skin-PL10', 
    'portraits/mediumskin-PM11': 'portraits/medium-skin-PM11', 
    'seasons/autumn-TM01': 'seasons/autumn_TM01', 
    'seasons/summer-SM01': 'seasons/summer_SM01', 
    'seasons/winter-WN01': 'seasons/winter_WN01', 
    'style/bw01': 'style/bw_BW01', 
    'style/film-inspired-boldbw': 'style/film_inspired_boldbw', 
    'style/film-inspired-coolbw': 'style/film_inspired_coolbw', 
    'style/film-inspired-warmgold': 'style/film_inspired_warmgold', 
    'style/futuristic-FT01': 'style/futuristic_FT01', 
    'style/vintage-VN01': 'style/vintage_VN01', 
    'subject/landscape-LN01': 'subject/landscape_LN01', 
    'subject/travel-TR01': 'subject/travel_TR01', 
    'subject/travel2-TR11': 'subject/travel2_TR11', 
    'subject/urbanarchitecture-UA08': 'subject/urban-architecture-UA08'}

tw_preset_naming ={
    'adaptive/blurbg-strong': 'adaptive/blurbg_strong', 
    'adaptive/blurbg-subtle': 'adaptive/blurbg_subtle', 
    'adaptive/enhance-portait': 'adaptive/enhance_portrait', 
    'adaptive/sky-bluedrama': 'adaptive/sky_bluedrama', 
    'adaptive/subject-pop': 'adaptive/subject_pop', 
    'portraits/lightskin-PL10': 'portraits/light-skin-PL10', 
    'portraits/mediumskin-PM11': 'portraits/medium-skin-PM11', 
    'style/bw01': 'style/bw_BW01', 
    'style/film-inspired-boldbw': 'style/film_inspired_boldbw', 
    'style/film-inspired-coolbw': 'style/film_inspired_coolbw', 
    'style/film-inspired-warmgold': 'style/film_inspired_warmgold', 
    'style/futuristic-FT01': 'style/futuristic_FT01', 
    'style/vintage-VN01': 'style/vintage_VN01', 
    'subject/landscape-LN01': 'subject/landscape_LN01', 
    'subject/travel-TR01': 'subject/travel_TR01', 
    'subject/travel2-TR11': 'subject/travel2_TR11', 
    'subject/urbanarchitecture-UA08': 'subject/urban-architecture-UA08'}


SOCIAL_PRESET_NAMING = {
    'facebook': fb_preset_naming,
    'telegram': tl_preset_naming,
    'twitter': tw_preset_naming,
}

def fix_preset_name(name, social=None):
    """Map a social-style preset name (e.g. 'seasons/autumn-TM01') to the
    tf_dataset name the checkpoint was trained under ('seasons/autumn_TM01').
    Returns the name unchanged if no --social flag is given or there is no entry."""
    if not name or not social:
        return name
    naming = SOCIAL_PRESET_NAMING.get(str(social).lower(), {})
    # print(f"naming: {naming}")
    return naming.get(name.strip('/'), name)
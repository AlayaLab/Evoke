import math


def resolve_trial(options, session_id):
    if not isinstance(options, dict) or options.get('enabled') is not True:
        return None
    if options.get('sessionId') != session_id:
        return None
    if options.get('mode') != 'fresh_history_low_coverage':
        raise ValueError('Unknown cameraControlTrial mode')
    threshold = options.get('coverageThreshold', .5)
    maximum = options.get('maxRecoveries', 32)
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not .05 <= threshold <= .75):
        raise ValueError('coverageThreshold must be finite and within .05–.75')
    if type(maximum) is not int or not 1 <= maximum <= 256:
        raise ValueError('maxRecoveries must be an integer within 1–256')
    return {'mode': options['mode'], 'coverageThreshold': float(threshold),
            'maxRecoveries': maximum, 'minimumTranslationSpan': .25}


def should_recover(options, count, chunk, pending_chunk, coverage, translation_span):
    if options is None or chunk < 2 or pending_chunk != chunk - 1:
        return False
    if count >= options['maxRecoveries']:
        return False
    if coverage is None or not math.isfinite(coverage) or not 0 <= coverage <= 1:
        return False
    return (math.isfinite(translation_span)
            and translation_span >= options['minimumTranslationSpan']
            and coverage < options['coverageThreshold'])

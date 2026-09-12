FRAME_RING_LIMIT = 96


def playback_settings(warmup_chunks=0, ahead_chunks=None, legacy_lead=84):
    if type(warmup_chunks) is not int or not 0 <= warmup_chunks <= 6:
        raise ValueError('Hidden warmup must be between 0 and 6 chunks.')
    if ahead_chunks is not None and (type(ahead_chunks) is not int or ahead_chunks != 1):
        raise ValueError('Interactive mode supports generating the next chunk while playing the current one.')
    if ahead_chunks is None:
        if type(legacy_lead) is not int or legacy_lead not in (60, 72, 84):
            raise ValueError('Invalid legacy playback lead')
        lead, initial, recovery, limit = legacy_lead, 72, 60, 96
    else:
        lead, initial, recovery, limit = 36, 36, 1, 72
    return dict(warmupChunks=warmup_chunks, aheadChunks=ahead_chunks,
                generationLeadFrames=lead, initialBufferFrames=initial,
                recoveryBufferFrames=recovery, publicationLimitFrames=limit)

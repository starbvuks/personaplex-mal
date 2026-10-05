"""Validated intervention policies; empty policy reproduces legacy computation."""
import math

DEFAULT_POLICY = dict(skip_start=True, write_enabled=True, read_scale=1.,
                      update_max_norm=.1, residual_max_ratio=.2, rates_mode='learned',
                      forget=.0003, momentum=.88, learning_rate=.0067,
                      write_source='legacy', write_prefix=True, feature_rms_norm=False,
                      state_update_max_norm=0.)
BOUNDS = {'read_scale':(0,2), 'update_max_norm':(0,100), 'residual_max_ratio':(0,10),
          'forget':(0,1), 'momentum':(0,.999), 'learning_rate':(0,1), 'state_update_max_norm':(0,100)}

# Explicit opt-in. These are initial experiment settings, not proven thresholds.
RESEARCH_POLICY = {**DEFAULT_POLICY, 'write_source':'observed', 'write_prefix':False,
                   'feature_rms_norm':True, 'update_max_norm':0., 'state_update_max_norm':.1}


def validate_policy(policy):
    if not isinstance(policy,dict) or set(policy)-set(DEFAULT_POLICY):
        raise ValueError('Unknown memory policy field')
    for key,value in policy.items():
        if key in ('skip_start','write_enabled','write_prefix','feature_rms_norm'):
            if type(value) is not bool: raise ValueError(f'{key} must be boolean')
        elif key=='rates_mode':
            if value not in ('learned','manual'): raise ValueError('Unknown rate mode')
        elif key=='write_source':
            if value not in ('legacy','observed'): raise ValueError('Unknown write source mode')
        else:
            low,high=BOUNDS[key]
            if type(value) not in (float,int) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{key} must be finite and between {low} and {high}')
    return dict(policy)


LEGACY_DEFAULTS = {**DEFAULT_POLICY, 'skip_start': False, 'update_max_norm': 0., 'residual_max_ratio': 0.}


def resolved_policy(policy):
    """Report actual fallback values, without changing the saved sparse policy."""
    return {**LEGACY_DEFAULTS, **validate_policy(policy)}


UNSET = object()


def resolve_policies(count, saved=None, requested=UNSET):
    """One contract for training, resume, microscope and evaluation.

    Empty/missing legacy policies remain empty: do not backfill modern caps.
    Explicit requests replace (rather than merge) every layer's policy.
    Saved data is validated even when an override was requested.
    """
    if saved is None:
        policies = [dict(DEFAULT_POLICY) for _ in range(count)]
        source = 'new_default'
    elif 'policies' not in saved:
        policies = [{} for _ in range(count)]
        source = 'legacy_checkpoint_missing'
    else:
        raw = saved['policies']
        if not isinstance(raw, list) or len(raw) != count:
            raise ValueError('Saved policies must contain exactly one policy per MAL layer')
        policies = [validate_policy(p) for p in raw]
        source = 'checkpoint'
    provenance = {'source': source, 'source_policies': [dict(p) for p in policies]}
    if requested is not UNSET:
        policy = validate_policy(requested)
        policies = [dict(policy) for _ in range(count)]
        provenance.update(source='explicit_override', inherited_from=source,
                          changed=policies != provenance['source_policies'])
    return policies, provenance


def apply_policies(branches, saved=None, requested=UNSET):
    policies, provenance = resolve_policies(len(branches), saved, requested)
    for branch, policy in zip(branches, policies):
        branch.policy = policy
        branch.policy_provenance = dict(provenance)
    return policies, provenance


def balanced_groups(rows):
    """Complete answer-balanced updates; no heldout row may enter training."""
    groups={}
    for row in rows:
        if row['split']!='train': continue
        groups.setdefault(row['answer'],[]).append(row)
    if not groups: raise ValueError('No training examples')
    if len({len(v) for v in groups.values()})!=1: raise ValueError('Balanced updates require equal counts per answer')
    return [list(items) for items in zip(*groups.values())]

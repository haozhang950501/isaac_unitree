from action_provider.action_provider_ces_grasp import CESGraspActionProvider
from action_provider.action_provider_dds import DDSActionProvider
from action_provider.action_provider_replay import FileActionProviderReplay
from action_provider.action_provider_wh_dds import DDSRLActionProvider


def create_action_provider(env, args):
    """create action provider based on parameters"""
    if args.action_source == "dds":
        return DDSActionProvider(env=env, args_cli=args)
    elif args.action_source == "dds_wholebody":
        return DDSRLActionProvider(env=env, args_cli=args)
    elif args.action_source == "ces_grasp":
        return CESGraspActionProvider(env=env, args_cli=args)
    elif args.action_source == "replay":
        return FileActionProviderReplay(env=env, args_cli=args)
    else:
        print(f"unknown action source: {args.action_source}")
        return None
from action_provider.action_provider_dds import DDSActionProvider
from action_provider.action_provider_replay import FileActionProviderReplay
from action_provider.action_provider_wh_dds import DDSRLActionProvider


def create_action_provider(env, args):
    """根据启动参数创建动作提供器；CES 仅在选中时延迟导入。"""
    if args.action_source == "dds":
        return DDSActionProvider(env=env, args_cli=args)
    elif args.action_source == "dds_wholebody":
        return DDSRLActionProvider(env=env, args_cli=args)
    elif args.action_source == "ces_grasp":
        # 延迟导入避免普通任务启动时加载 CES 场景、状态机与 IK 依赖。
        from action_provider.action_provider_ces_grasp import CESGraspActionProvider

        return CESGraspActionProvider(env=env, args_cli=args)
    elif args.action_source == "replay":
        return FileActionProviderReplay(env=env, args_cli=args)
    else:
        print(f"unknown action source: {args.action_source}")
        return None

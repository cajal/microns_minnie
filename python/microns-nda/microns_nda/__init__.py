from microns_utils import version_utils

__version__ = version_utils.check_package_version(
    package='microns-nda', 
    check_if_latest=True, 
    check_if_latest_kwargs=dict(
        owner='cajal', 
        repo='microns-nda', 
        source='tag', 
    )
)

def check_latest_version_from_github(owner='cajal', repo='microns-nda', source='tag', branch=None, path_to_version_file=None, warn=True):
    """
    Wrapper for :func:`~microns_utils.version_utils.check_latest_version_from_github`
    """
    return version_utils.check_latest_version_from_github(owner=owner, repo=repo, source=source, branch=branch, path_to_version_file=path_to_version_file, warn=warn)
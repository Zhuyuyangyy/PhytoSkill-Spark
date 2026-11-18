"""AgentShield NLIP Security Extension (SafeOSE '27 hackathon, Areas 03 and 06).

Requires the optional ``nlip`` extra: ``pip install -e ".[nlip]"``.
"""

from shield.nlip.delegation import (AgentIdentity, DelegationError, RevocationList, TrustStore,
                                    issue, verify_chain)
from shield.nlip.extension import (AgentShieldSE, Decision, SEPolicy, attach_delegation,
                                   attach_tool_call, seal)

__all__ = ["AgentIdentity", "AgentShieldSE", "Decision", "DelegationError", "RevocationList",
           "SEPolicy", "TrustStore", "attach_delegation", "attach_tool_call", "issue", "seal",
           "verify_chain"]

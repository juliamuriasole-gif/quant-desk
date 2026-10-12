"""Errors raised when the desk constitution is broken."""


class ConstitutionViolation(Exception):
    """An action that breaks the desk constitution."""


class RoleViolation(ConstitutionViolation):
    """An agent was asked to do a task outside its assigned role."""

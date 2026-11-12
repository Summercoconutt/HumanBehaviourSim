# Repository visibility check

**Issue:** #5  
**Checked:** 12 November 2026

The GitHub repository API reports this repository as **public**. The working agreement described in issue #5 expects a private repository.

A search of the current tracked text files found no credential, token, or private-key patterns. `Data/final.csv` is the car-evaluation table plus two extra columns, and that dataset is already described in the public proposal. A full history-and-fork review is still an owner security step if the repository has ever been public.

Changing visibility is an owner action: GitHub Settings, General, Danger Zone, Change repository visibility. This check does not switch the repository to private, because issue #5 says to confirm the required policy with the project owner before that change. After the owner makes it private, access should be re-checked from a logged-out session, and any public forks should be reviewed.

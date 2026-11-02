# RQ2 demonstration and remaining checks

The prototype demonstrates how AI evidence can inform SIEM, SOC and SOAR-style obligations within the four-layer architecture. The identity and network layers supply simulated context. The detection layer supplies a fixed model prediction and quality flags. The response layer returns recommendations using exploratory V4 and never executes an action.

1. Run the included sample in **Predict a CSV** and retain its prediction CSV and run JSON.
2. Open **RQ2 decision support**, select **Latest prediction run** and evaluate a record. Explain which fields came from inference and which were entered as simulated context.
3. Choose an all-zero sample record. Explain why source evidence is requested without claiming that its supplied label is wrong.
4. Replay illustrative N27 and N28. Explain why a benign prediction or unusable model evidence does not remove approval obligations for a high-impact proposal accompanied by a secondary signal.
5. Replay N12. The view can recommend read-only evidence collection under its simulated conditions. It still records execution_authorised=False.
6. Download the RQ2 decision JSON and CSV. Compare their record ID, policy inputs, reasons and obligations. They identify the evaluated decision; editing an unsubmitted form does not update that recorded decision.
7. Explain the handoff: SIEM retains/correlates evidence, SOC investigates, and SOAR would coordinate a separately authorised playbook. The prototype contains no live platform connector or response executor.

These examples were available during V4 repair. Their replay and the app tests are integration/regression evidence. They do not replace the frozen V3 evaluation, independently validate V4, measure a safe automation percentage or demonstrate reduced analyst workload.

## Public demonstration still to verify

- Confirm the exact app/deployment URL and the source commit it serves.
- Open it signed out with the intended assessor access conditions.
- Repeat prediction, invalid-input rejection, historical review and RQ2 JSON/CSV downloads in the deployed app.
- Check it from another device/network and complete any required D633 demonstration.
- Record actual outcomes, dates and failures in the cloud test sheet. Do not mark a source-code or local-interface test as a public-cloud pass.

The previously recorded URL showed an access-or-missing-app page in a signed-out browser on 2 November 2026. Owner-side deployment/access verification is still needed.

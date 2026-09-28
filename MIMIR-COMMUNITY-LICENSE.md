# MIMIR Community Model License

**Version 1.0 — Draft for legal review before production use**

Copyright © 2026 Mythologic. All rights reserved except as expressly licensed below.

> **Important:** This is a proprietary model license. It is **not** an OSI-approved open-source license, and the MIMIR weights are not released as “open source”. The public MIMIR software/SDK may be separately licensed under Apache License 2.0. Do not replace this file with an open-source license for the model weights unless the business and regulatory strategy is intentionally changed.

## 1. Purpose and scope

This MIMIR Community Model License (the “**License**”) governs the use of the MIMIR model identified in the applicable distribution page, including any model weights, tokenizer/model configuration files, inference artifacts, quantized versions, converted versions, and other model files expressly identified as licensed under this License (collectively, the “**Model**”).

The Model is a non-generative decision and routing model intended to return structured decisions from structured or unstructured input. The Model may be used with the public MIMIR software, with other software, or independently, subject to this License.

This License does **not** grant access to, license, disclose, or authorize use of any non-public MIMIR source code, training code, training data, data-collection pipeline, private evaluation infrastructure, internal research materials, proprietary training methodology, private checkpoints, internal documentation, trade secrets, or other non-public materials of Licensor (collectively, “**Non-Public MIMIR Materials**”).

## 2. Definitions

### 2.1 “Licensor”

“Licensor” means Mythologic, and any successor entity that lawfully owns or controls the rights in the Model.

### 2.2 “You” and “Licensee”

“You” and “Licensee” mean the natural person or legal entity exercising rights under this License.

If a natural person uses the Model on behalf of an organization, employer, client, principal, customer, affiliate, or other beneficiary, that organization or beneficiary is the Licensee for that use. A person may use the Model as an individual only for a genuinely personal project that is not performed for, paid for by, controlled by, or intended to benefit an organization.

### 2.3 “Organization”

“Organization” means any company, partnership, association, nonprofit entity, public body, academic institution, governmental entity, unincorporated business, or other legal or economic organization, together with its Controlled Affiliates when assessing eligibility under this License.

### 2.4 “Controlled Affiliate”

A “Controlled Affiliate” is an entity that directly or indirectly controls, is controlled by, or is under common control with the Licensee. “Control” means ownership or voting power of more than fifty percent (50%), the contractual right to appoint a majority of governing persons, or equivalent practical control.

### 2.5 “Economic Group”

The “Economic Group” is the Licensee and all Controlled Affiliates, together with any parent entity that controls the Licensee and entities controlled by that parent.

For purposes of eligibility, the Economic Group is treated as a single organization. A reorganization, subsidiary, special-purpose vehicle, contractor, reseller, nominee, employee, individual account, or other intermediary may not be used to avoid the revenue threshold or any other restriction in this License.

### 2.6 “Community-Eligible Licensee”

A Licensee is “Community-Eligible” when all of the following are true for the applicable use:

1. The Licensee is an individual acting on the individual's own behalf, **or** an Organization whose Economic Group satisfies the Revenue Test below.
2. The use is not on behalf of, for the benefit of, or controlled by a non-eligible Organization.
3. The Licensee is not using an intermediary to conceal the identity or economic beneficiary of the actual user.
4. The Licensee complies with this License.

### 2.7 “Revenue Test”

An Organization passes the Revenue Test when its Economic Group's aggregate annual revenue does not exceed **US$500,000 (five hundred thousand United States dollars)** or the equivalent amount in the applicable reporting currency.

For an Organization with an established fiscal year, the primary test is the greater of:

- consolidated revenue for the immediately preceding completed fiscal year; and
- consolidated revenue for the immediately preceding twelve-month period for which reliable financial information is available.

For an Organization without a completed fiscal year, the test is based on the Organization's actual consolidated revenue during the first available twelve-month period, or, where fewer than twelve months have elapsed, the annualized revenue reasonably supported by actual results. Once actual or annualized revenue exceeds the threshold, the Organization must transition to a Commercial License as provided in Section 11.

Revenue is measured before ordinary operating expenses and excludes VAT, sales taxes, and amounts eliminated on consolidation. Revenue includes ordinary sales, licensing, subscriptions, services, usage fees, and other business operating revenue. Grants, donations, and non-commercial public funding are excluded unless they are received as consideration for commercial products or services. A natural person carrying on a sole proprietorship, freelance, consulting, or other business activity is treated as an Organization for this Revenue Test with respect to that business activity. Personal non-business use by that person is not aggregated with the person's unrelated business revenue unless the Model is used for or benefits the business.

Revenue from Controlled Affiliates and the parent Economic Group is aggregated. A transfer of an activity to another entity within the same Economic Group does not reset the threshold.

Where an Organization reports in a currency other than USD, the USD equivalent is determined using the European Central Bank reference exchange rate applicable on the final business day of the relevant reporting period, or another objectively verifiable official central-bank rate if no ECB reference rate is available for the currency concerned.

### 2.8 “Commercial License”

A “Commercial License” is a separate paid written agreement executed between Licensor and the applicable Organization, including any order form or online commercial agreement expressly identified as a Commercial License.

### 2.9 “Evaluation Use”

“Evaluation Use” means temporary internal testing by a non-Community-Eligible Organization solely to determine whether to obtain a Commercial License. Evaluation Use:

- lasts no more than thirty (30) consecutive calendar days from first access;
- may not be used in production;
- may not be used to provide a service to third parties;
- may not be used to make or distribute a commercial fine-tune or derivative model;
- may not be used to avoid a Commercial License; and
- does not include fine-tuning the Model, except with Licensor's prior written authorization.

Evaluation Use is a limited exception to the commercial-use restrictions and does not grant any continuing rights after the evaluation period.

### 2.10 “MIMIR-Derived Model”

A “MIMIR-Derived Model” means any model, checkpoint, weight set, parameter set, quantized artifact, pruned artifact, merged artifact, converted artifact, or other model representation that incorporates, adapts, fine-tunes, initializes from, merges with, or is otherwise materially derived from Model weights.

An adapter, LoRA, prompt-tuning artifact, or other auxiliary artifact that does not contain Model weights is an “**Adapter Artifact**”.

### 2.11 “Competitive Model”

A “Competitive Model” means a model or model service whose principal commercial purpose is to substitute for, replicate, imitate, or materially reproduce the core decision-routing, classification, scoring, verification, ranking, estimation, or structured decision capabilities of MIMIR using the Model, MIMIR outputs, or protected MIMIR internal signals as a material source.

Ordinary benchmarking, interoperability testing, academic research, and independent model development that does not materially use MIMIR as training or distillation data are not, by themselves, Competitive Model development.

### 2.12 “Production Use”

“Production Use” means use in a live service, customer-facing product, revenue-generating workflow, operational decision system, internal business process relied upon for material business operations, or any other deployment beyond temporary evaluation or development.

### 2.13 “Distribute”

“Distribute” means to provide, publish, transfer, sell, sublicense, host for third-party access, package, embed, or otherwise make the Model or a MIMIR-Derived Model available to another person or entity, whether for consideration or not.

## 3. Acceptance and electronic contracting

This License is a contract between Licensor and Licensee to the extent permitted by applicable law. You accept this License by any of the following acts after being presented with or having reasonable access to these terms: downloading the Model, accessing or loading the Model, installing or executing software that incorporates the Model, fine-tuning the Model, Distributing the Model or a MIMIR-Derived Model, or otherwise exercising any right granted by this License.

For commercial Organizations, continued use after any applicable Evaluation Use period requires a separate Commercial License. Licensor may require an Organization to complete an explicit click-through acceptance, purchase process, or signed agreement before granting commercial rights.

If You do not agree to this License, You must not download, use, copy, modify, fine-tune, Distribute, or otherwise exercise rights in the Model, except to the extent a mandatory rule of law independently grants a non-waivable right.

## 4. Community license grant

Subject to continued compliance with this License, Licensor grants a worldwide, non-exclusive, royalty-free, non-transferable license to a Community-Eligible Licensee to:

1. download and possess the Model;
2. execute and use the Model for development, research, evaluation, or Production Use;
3. use the Model in personal projects and commercial products, provided the Licensee remains Community-Eligible;
4. fine-tune, adapt, quantize, prune, merge, convert, optimize, or otherwise modify the Model;
5. create and use MIMIR-Derived Models for the Licensee's own permitted purposes;
6. create and use Adapter Artifacts; and
7. Distribute the unmodified Model and MIMIR-Derived Models, subject to Sections 7 and 8.

This grant is a limited license. It is not a sale or transfer of ownership of the Model or of Licensor's intellectual property.

## 5. Commercial organizations

An Organization that does not pass the Revenue Test may not exercise Community rights for any purpose except Evaluation Use unless it first obtains a Commercial License.

Once an Organization no longer passes the Revenue Test, Community rights end thirty (30) calendar days after the Organization becomes aware, or reasonably should have become aware, that the threshold has been exceeded, unless a Commercial License is executed earlier.

A Commercial License is required for the Organization's continued use, Production Use, internal business use, fine-tuning, creation or use of MIMIR-Derived Models, hosting, API service, or Distribution after that transition period.

The revenue threshold applies to the actual economic beneficiary of the use. It does not matter whether the Model is obtained through an employee, founder, consultant, contractor, freelancer, reseller, affiliate, university account, public repository, Hugging Face account, GitHub account, personal computer, cloud account, or other intermediary.

## 6. Personal use by employees and contractors

An employee, officer, contractor, student, freelancer, or consultant may use the Model under the Community License only when acting genuinely on that person's own behalf and not for, at the direction of, with the resources of, or for the benefit of a non-Community-Eligible Organization.

Use of the Model on employer or client data, infrastructure, credentials, repositories, accounts, customer workloads, commercial deliverables, or other resources is deemed use on behalf of that employer or client.

A contractor cannot provide Community-Licensed MIMIR use to a non-Community-Eligible Organization as a workaround for obtaining a Commercial License.

## 7. Distribution and derivative models

### 7.1 Notice requirement

You may Distribute the Model or a MIMIR-Derived Model only if You:

- include a copy of this License or a durable link to the exact version of this License;
- preserve all copyright and attribution notices;
- clearly identify the model as MIMIR or as a derivative of MIMIR where applicable;
- state meaningful modifications, including material fine-tuning or quantization, where applicable; and
- do not represent the derivative as an official MIMIR release unless Licensor has expressly authorized that statement.

### 7.2 Same-license requirement for weights

A distributed MIMIR-Derived Model containing or embodying MIMIR weights must remain subject to this License or a later Licensor-issued license expressly designated by Licensor as a successor license for that model version.

You may license an Adapter Artifact under other terms only where the Adapter Artifact does not contain MIMIR weights and is clearly distributed as an adapter requiring a separately obtained MIMIR base model.

### 7.3 No sublicensing around the commercial boundary

You may not grant, convey, sublicense, or purport to grant another person rights that You do not possess under this License. In particular, You may not use Distribution through an individual, small entity, contractor, reseller, or intermediary to grant a non-Community-Eligible Organization rights that would require a Commercial License.

### 7.4 No bundling for commercial organizations

A Community-Eligible Licensee may not bundle the Model or a MIMIR-Derived Model into a product, service, image, appliance, hosted environment, or other offering supplied specifically for the use of a non-Community-Eligible Organization without a Commercial License covering that Organization.

## 8. Fine-tuning and model extraction

Community-Eligible Licensees may fine-tune the Model for permitted purposes.

You may not use the Model, its internal representations, hidden states, logits, gradients, weights, or MIMIR outputs as a material training, distillation, imitation, or model-extraction source for a Competitive Model, except for ordinary benchmarking and interoperability testing.

This restriction does not prohibit independent implementation of general ideas, mathematical concepts, or techniques that are independently developed without using protected MIMIR materials, subject to applicable law.

You may not intentionally perform or facilitate systematic model extraction, weight extraction, parameter recovery, or other technical processes primarily intended to reconstruct the Model or materially circumvent this License.

## 9. No access to proprietary training technology

The Model is distributed without a license to any Non-Public MIMIR Materials.

Nothing in this License grants rights to:

- MIMIR architecture source code;
- MIMIR training source code;
- proprietary datasets or data licenses;
- private training recipes or optimization schedules;
- unpublished experiments or ablations;
- private checkpoints;
- proprietary evaluation infrastructure; or
- Licensor's trade secrets or confidential information.

Publication of a model configuration, model card, benchmark result, or other information about the Model does not constitute a license to reproduce Licensor's unpublished training system.

## 10. Reverse engineering and mandatory rights

You may not decompile, disassemble, reverse engineer, reconstruct, extract source code from, or otherwise seek to derive the proprietary implementation or training system of the Model, except to the extent a mandatory rule of applicable law expressly permits or requires such activity and the relevant statutory conditions are satisfied.

Nothing in this License is intended to restrict a non-waivable legal right, including legally permitted acts required for interoperability, security testing, software observation, study, or other mandatory statutory exceptions.

Where applicable law grants a mandatory right that conflicts with this Section, that mandatory right prevails only to the extent of the conflict.

## 11. Revenue threshold, change of status, and transition

You are responsible for determining whether You remain Community-Eligible.

A Community-Eligible Organization must maintain reasonably sufficient records to substantiate its eligibility for the preceding and applicable current periods.

The Organization must notify Licensor promptly when:

- it exceeds the Revenue Test threshold;
- its parent or Controlled Affiliate changes such that the Economic Group exceeds the threshold;
- it undergoes a merger, acquisition, or change of control that causes the Economic Group to exceed the threshold; or
- it begins a use that requires a Commercial License.

Following a threshold event, the Organization receives a thirty (30) day transition period to cease restricted use or execute a Commercial License. The transition period does not authorize new Production Use, new Distribution, or new fine-tuning after the threshold event.

Use that occurred lawfully before the threshold event is not retroactively converted into a breach merely because the Organization later crossed the threshold.

## 12. Verification and audit

Upon reasonable written notice, Licensor may request information reasonably necessary to verify compliance with the Community License, including organizational identity, Economic Group structure, Revenue Test information, and description of material uses of the Model.

Licensee must provide reasonably sufficient supporting information within fifteen (15) business days after a request when the request is made in good faith and is proportionate to a reasonable compliance concern.

If Licensor has documented reason to believe that material non-compliance has occurred and Licensee does not resolve the concern or provide reasonable evidence of compliance, Licensor may conduct a compliance audit through an independent auditor subject to reasonable confidentiality obligations. Audits will be limited to information reasonably relevant to this License, conducted during normal business hours, and designed to minimize operational disruption.

Licensor will bear ordinary audit costs unless an audit establishes a material breach, in which case Licensee will reimburse reasonable audit costs.

## 13. Prohibited circumvention

You may not directly or indirectly circumvent, defeat, avoid, or materially frustrate:

- the Revenue Test;
- the requirement for a Commercial License;
- the prohibition on non-eligible Organizations using Community rights;
- the restrictions on MIMIR-Derived Models;
- the restrictions on Competitive Models; or
- Licensor's reasonable compliance-verification process.

Circumvention includes, without limitation, using nominees, employees, contractors, shell companies, affiliates, resellers, personal accounts, educational accounts, public demos, or third-party services when the actual economic beneficiary is a non-Community-Eligible Organization.

## 14. Attribution

You must retain the following notice in any copy of the Model or MIMIR-Derived Model that You Distribute:

> “This product includes or is derived from MIMIR, a model developed by Mythologic. MIMIR is provided under the MIMIR Community Model License. See the applicable LICENSE.md for terms.”

You are not required to display the notice on every end-user screen or in every generated output merely because the Model is used internally.

## 15. Trademark and branding

This License grants no right to use the MIMIR, Mythologic, logos, product names, or other trademarks of Licensor except as reasonably necessary to make accurate factual statements about compatibility or derivation.

You may state that an application “uses MIMIR” or is “compatible with MIMIR” when that statement is accurate and not misleading. You may not use Licensor's marks in a manner that implies endorsement, certification, partnership, sponsorship, affiliation, or official status without prior written permission.

See `TRADEMARKS.md` for the applicable brand policy.

## 16. Third-party components and separate rights

The Model may include, depend on, or be distributed alongside third-party components, data, tokenizers, libraries, or assets subject to separate licenses.

Those separate terms remain applicable to the relevant components. You are responsible for complying with them.

This License does not purport to grant rights to third-party intellectual property that Licensor does not own or control.

## 17. Compliance with law

You are solely responsible for ensuring that your use of the Model complies with all applicable laws and regulations, including intellectual property, privacy, data protection, consumer protection, export controls, sanctions, sector-specific requirements, employment law, financial regulation, medical regulation, and applicable artificial-intelligence regulation.

The Model may be capable of being used in regulated or high-impact contexts. This License does not constitute legal or regulatory authorization to deploy it in any particular sector or use case.

Where applicable law requires documentation, risk management, transparency, human oversight, monitoring, or other controls, Licensee is responsible for implementing those controls for its use of the Model.

## 18. No warranties

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE MODEL IS PROVIDED “AS IS” AND “AS AVAILABLE”, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, ACCURACY, RELIABILITY, AVAILABILITY, OR THAT THE MODEL WILL MEET ANY PARTICULAR PERFORMANCE OR REGULATORY REQUIREMENT.

Licensor does not warrant that outputs are accurate, complete, lawful, safe, non-infringing, or suitable for any particular decision.

Licensee is responsible for validating outputs, applying appropriate human oversight, and determining whether the Model is suitable for the intended use.

## 19. Limitation of liability

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, LICENSOR AND ITS AFFILIATES, OFFICERS, EMPLOYEES, CONTRIBUTORS, AND LICENSORS WILL NOT BE LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, CONSEQUENTIAL, OR PUNITIVE DAMAGES, OR FOR LOST PROFITS, LOST REVENUE, LOSS OF DATA, LOSS OF GOODWILL, BUSINESS INTERRUPTION, OR SUBSTITUTE PROCUREMENT ARISING FROM OR RELATING TO THE MODEL OR THIS LICENSE.

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, LICENSOR'S AGGREGATE LIABILITY ARISING OUT OF OR RELATING TO THIS LICENSE OR THE MODEL WILL NOT EXCEED THE GREATER OF (A) US$100 OR (B) THE FEES ACTUALLY PAID TO LICENSOR BY LICENSEE FOR THE MODEL OR A COMMERCIAL LICENSE DURING THE TWELVE (12) MONTHS PRECEDING THE EVENT GIVING RISE TO THE CLAIM.

Nothing in this License limits liability that cannot lawfully be limited, including liability that applicable law makes non-excludable.

## 20. Indemnification

To the maximum extent permitted by applicable law, Licensee will defend, indemnify, and hold harmless Licensor and its affiliates, officers, employees, and contributors from third-party claims and reasonable costs arising from Licensee's unlawful use of the Model, breach of this License, unauthorized Distribution, violation of third-party rights caused by Licensee's modifications or combination of the Model, or use of the Model in a regulated or high-impact context without legally required controls.

This Section does not require Licensee to indemnify Licensor to the extent a claim is caused solely by Licensor's fraud, willful misconduct, or a non-waivable legal obligation of Licensor.

## 21. Term and termination

This License begins when accepted and continues until terminated.

Licensor may terminate this License as to You if You materially breach its terms and fail to cure the breach within fifteen (15) calendar days after written notice, except that Licensor may terminate immediately for breaches that are incapable of cure, deliberate circumvention, unauthorized commercial use by a non-Community-Eligible Organization, unauthorized Distribution that materially harms Licensor's rights, or violation of applicable law.

Upon termination, You must promptly stop exercising rights under this License, stop distributing the Model and MIMIR-Derived Models, and delete or return copies of the Model and MIMIR-Derived Models in Your possession or control, except where retention is required by law or a backup system created automatically in the ordinary course, provided retained copies are not restored or used.

Outputs independently created before termination need not be deleted solely because this License terminated, unless another law or contractual obligation requires deletion.

Sections concerning ownership, restrictions, confidentiality, compliance, liability, indemnity, dispute resolution, and any provision that by its nature should survive will survive termination.

## 22. Ownership

Licensor and its licensors retain all right, title, and interest in and to the Model and Licensor-owned intellectual property except for the limited rights expressly granted by this License.

No ownership interest is transferred to Licensee.

Licensee retains ownership of its own data, prompts, inputs, and independent software, subject to third-party rights and applicable law.

## 23. Feedback

If You voluntarily provide suggestions, bug reports, benchmark results, or other feedback about MIMIR, You grant Licensor a perpetual, worldwide, irrevocable, royalty-free, transferable, sublicensable license to use that feedback for any purpose, including improving, commercializing, and developing MIMIR, without an obligation to compensate You. Licensor will not be required to treat ordinary feedback as confidential.

Do not submit confidential information, personal data, or third-party proprietary information as feedback unless You have the right to do so.

## 24. Versioning and no retroactive expansion

Each MIMIR model release is governed by the license version identified for that release.

Licensor may publish future model releases under different terms. Unless expressly permitted by a separate agreement, a change to the license for a future release does not retroactively alter the terms applicable to a previously obtained release.

Licensor may publish a successor version of this License for future releases and may expressly designate whether existing releases can migrate to the successor version.

## 25. No implied rights

Except for the rights expressly granted in this License, no rights are granted by implication, estoppel, exhaustion, or otherwise.

## 26. Assignment

Licensee may not assign this License independently of the Model or delegate its obligations without Licensor's prior written consent, except to a legal successor in a bona fide merger or sale of substantially all relevant assets, provided the successor assumes this License and remains compliant with the applicable eligibility rules.

Any change of control that causes the Licensee's Economic Group to cease to be Community-Eligible is treated as a threshold event under Section 11.

Licensor may assign this License and its rights in the Model to a successor or acquirer of the relevant intellectual property.

## 27. Governing law and dispute resolution

This License is governed by the laws of the European Union and, for Licensees domiciled in the United States, the laws of the United States, excluding conflict-of-law rules to the extent permitted by law. Subject to mandatory jurisdiction rules, disputes are heard by the competent courts of the Licensee's domicile within the EU or the US.

Before commencing proceedings, the parties will attempt in good faith to resolve the dispute through written notice and senior-level discussion for at least fifteen (15) calendar days, except where urgent injunctive or protective relief is reasonably necessary.

If the Licensor's final legal entity or principal place of business is outside the EU and the US, this Section must be reviewed and replaced by counsel before publication.

## 28. Mandatory law; consumer and non-waivable rights

Nothing in this License excludes, restricts, or waives any right, remedy, or protection that applicable law prohibits the parties from excluding or waiving.

If any provision is held invalid or unenforceable, it will be enforced to the maximum extent permitted by law and the remainder of this License will remain in effect.

## 29. Entire agreement

This License, together with any expressly incorporated license notices and the terms applicable to third-party components, constitutes the entire agreement concerning the Model between Licensor and Licensee concerning the subject matter hereof, except that a separate Commercial License or signed agreement controls where it expressly states that it supersedes this License.

No purchase order, procurement portal, click-through term, or vendor form submitted by Licensee will modify this License unless Licensor expressly accepts the modification in writing.

## 30. Contact

Legal notices and commercial licensing requests: contact Mythologic.

---

**End of MIMIR Community Model License v1.0**

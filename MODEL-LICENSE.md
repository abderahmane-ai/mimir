# MIMIR Model License

**Version 1.0.**

Copyright © 2026 Mythologic. All rights reserved except as expressly licensed below.

> **Not an open-source license.** The MIMIR Model License is a proprietary license for the
> MIMIR-1 model weights. The MIMIR SDK and software are licensed separately under the Apache
> License 2.0, which does not license the MIMIR model weights.

> **The short version** — for convenience only; the License text below is the contract.
>
> - **Free to use, including commercially**, if you are an individual acting for yourself or
>   an Organization under the Revenue Threshold (US$1,000,000 annual gross revenue):
>   production, modification, fine-tuning, quantization, derivatives, integration,
>   distribution, customer deployment, and paid products and services.
> - **No registration, no approval, no payment, no royalties.** Download MIMIR, use MIMIR,
>   build with MIMIR, ship MIMIR.
> - **Evaluating is free for everyone.** Any organization, at any revenue, may download,
>   inspect, benchmark, evaluate, prototype and test MIMIR internally without contacting
>   Mythologic.
> - **Crossing the Revenue Threshold does not terminate rights already granted.** What you
>   lawfully obtained while eligible keeps working — no shutdown countdown, no deletion, no
>   retroactive fee.
> - **A commercial agreement adds rights** — OEM, white-label, model-as-a-service, broad
>   redistribution, enterprise support, warranties, indemnification, and other negotiated
>   rights. See `COMMERCIAL-LICENSING.md`.
> - **Keep it lawful, do not train a competing model from MIMIR, and keep the attribution
>   notice when you distribute weights.**

## 1. Purpose and scope

This MIMIR Model License (the “License”) governs the MIMIR-1 model: its weights, the model
artifacts distributed with them, the tokenizer and model configuration files necessary to use
the weights, and any copy, conversion, quantization or other modification of those materials
(the “Model”). It also governs MIMIR-Derived Models as described in Section 8.

The Model is a non-generative decision model that returns structured decisions. This License
applies whether the Model is used through the MIMIR SDK or through other software.

The MIMIR SDK and other Mythologic software are licensed under the Apache License 2.0. That
license covers the software only. It does not license the Model or anything else governed by
this License.

This License does not grant access to Mythologic's private development assets (Section 9).

## 2. Definitions

**“Licensor”** means Mythologic, and any successor that lawfully holds the rights in the
Model.

**“You”** and **“Licensee”** mean the person or entity exercising rights under this License.
If You use the Model for an organization, that organization is the Licensee for that use.

**“Organization”** means any company, partnership, association, nonprofit, public body,
academic institution, governmental entity or other legal or economic organization.

**“Economic Group”** means the Licensee, every entity that controls it, every entity it
controls, and every entity under common control with it. “Control” means more than fifty
percent (50%) of the voting power or the practical ability to direct management. The Economic
Group is treated as one Organization under this License.

**“Revenue Threshold”** means US$1,000,000 in annual gross revenue. It is measured on a
consolidated basis across the Economic Group, for the most recently completed fiscal year, or
for the preceding twelve months where that figure is higher, before operating expenses and
excluding VAT and sales taxes. Where revenue is reported in another currency, convert it at a
published central-bank reference rate. A sole proprietorship, freelance or consulting business
is treated as an Organization for its business revenue.

**“Community-Eligible Licensee”** means an individual using the Model genuinely on the
individual's own behalf, or an Organization whose Economic Group does not exceed the Revenue
Threshold.

**“Evaluation”** means internal use of the Model to determine whether to adopt it: downloading,
inspecting, benchmarking, evaluating, prototyping with it, and internal technical testing.
Evaluation does not include Production Use.

**“Production Use”** means use in a live product, service, customer-facing application,
revenue-generating workflow, or business operation beyond Evaluation.

**“Distribute”** means to provide, publish, transfer, sell, sublicense, host for third-party
access, embed, or otherwise make the Model or a MIMIR-Derived Model available to another
person or entity, with or without consideration.

**“MIMIR-Derived Model”** means any model, weights, checkpoint, parameter set, quantization,
pruned, merged or converted artifact, or other model representation that incorporates,
adapts, fine-tunes, initializes from, or is otherwise materially derived from the Model's
weights. An adapter, LoRA or similar artifact that does not contain Model weights is an
“**Adapter Artifact**”.

**“Reserved Rights”** means the rights listed in Section 7.

**“Commercial Agreement”** means a separate written agreement with Licensor that grants
additional rights.

## 3. Community grant

If You are a Community-Eligible Licensee, Licensor grants You a worldwide, non-exclusive,
royalty-free, perpetual, and irrevocable license — subject to termination for breach — to:

1. use the Model, including for commercial use, production deployment, internal use,
   research and evaluation;
2. modify, fine-tune, quantize, optimize and otherwise adapt the Model;
3. create and use Adapter Artifacts and MIMIR-Derived Models;
4. integrate the Model into applications;
5. Distribute the Model and MIMIR-Derived Models under Section 8; and
6. deploy the Model for Your customers and use it in paid products and services.

No registration, application, approval, contact, payment or royalty is required for this
grant. Download MIMIR, use MIMIR, build with MIMIR, ship MIMIR.

The grant is a license, not a sale: Licensor retains ownership of the Model and its
intellectual property.

**Patent grant.** To the extent Licensor holds patents that cover the Model as delivered,
Licensor grants You a worldwide, royalty-free, non-transferable patent license to make, have
made, use and import the Model for the rights granted above. This patent license ends, as of
the date a lawsuit is filed, if You or Your affiliates institute patent litigation alleging
that the Model, a MIMIR-Derived Model or their use infringes a patent.

## 4. Conditions

The grant in Section 3 is subject to this License. These are the License's only restrictions.

### 4.1 Competing models

Do not use the Model, its weights, internal representations, hidden states, logits, gradients
or outputs as a material source for training, distilling, imitating or extracting a model
whose principal purpose is to substitute for MIMIR's decision capabilities. Ordinary
benchmarking, interoperability testing, academic research, independent development, using the
Model as a component of Your application, and fine-tuning under Section 3 are all permitted.

### 4.2 No avoidance through intermediaries

Do not use employees, contractors, consultants, nominees, resellers, affiliates, separate
accounts or a reorganization to avoid the Revenue Threshold, or to give an Organization that
is not Community-Eligible the benefit of the community grant.

### 4.3 No extraction to evade this License

Do not perform or facilitate systematic model extraction, weight extraction or parameter
recovery primarily intended to reconstruct the Model or to place the Model outside this
License.

### 4.4 Lawful use

Do not use the Model to violate applicable law or the rights of others. In particular, do not
use it:

- to deceive, defraud or harass people;
- to exploit or endanger children;
- to develop or operate weapons;
- to discriminate against people in housing, employment, credit, education or public
  accommodations in violation of applicable law; or
- to make consequential decisions about people without the human oversight applicable law
  requires.

You are responsible for complying with the laws that apply to Your use, including privacy,
export-control and AI-regulation requirements. The Model is distributed without personal
data; You are responsible for the lawfulness of the data You process with it.

### 4.5 Attribution

If You Distribute the Model or a MIMIR-Derived Model, You must retain this notice in the
materials You distribute:

> “This product includes or is derived from MIMIR, a model developed by Mythologic, and is
> provided under the MIMIR Model License. See the applicable LICENSE for terms.”

## 5. Evaluation is free for everyone

Anyone may download, inspect, benchmark, evaluate, prototype with, and conduct internal
technical testing of the Model without fee, registration or contacting Licensor, regardless
of revenue. Evaluation is not Production Use. An Organization that is not Community-Eligible
needs a Commercial Agreement before Production Use (Section 7).

## 6. Crossing the Revenue Threshold

This Section is the transition mechanism, and it is deliberately not a cliff.

1. **Existing lawful rights continue.** If You were Community-Eligible when You obtained a
   release, the grant in Section 3 is perpetual and irrevocable for that release and for
   MIMIR-Derived Models You create from it while eligible, even if You later cease to be
   Community-Eligible. You may continue to exercise every right in Section 3 in them,
   including new deployments, new products, new customers and continued Distribution. There
   is no shutdown countdown, no obligation to delete anything, and no retroactive fee.
2. **Prospective boundary.** The community grant does not extend to Model releases You first
   obtain after You cease to be Community-Eligible, and it never extends to the Reserved
   Rights.
3. **New or expanded enterprise rights** beyond the community grant are available under a
   Commercial Agreement (Section 7).
4. A merger, acquisition, reorganization or change of control does not retroactively
   invalidate rights already granted under this Section.

You are responsible for determining Your own eligibility. Licensor does not require
eligibility reports, registration or royalty statements.

## 7. Enterprise licensing and Reserved Rights

An Organization above the Revenue Threshold, and any Licensee that wants rights beyond the
community grant, may obtain a Commercial Agreement from Licensor. The rights that this
License does not grant, and that a Commercial Agreement may add (the “Reserved Rights”),
include:

- OEM rights;
- white-label redistribution;
- broad redistribution rights — offering the Model, or a substantially unmodified copy of
  it, to third parties as a model product, beyond embedding it in Your own applications;
- model-as-a-service rights;
- extensive affiliate rights;
- contractual enterprise deployment rights;
- custom fine-tuning and model-development services;
- enterprise support and service-level agreements;
- warranties and indemnification;
- security and compliance commitments; and
- special deployment or distribution arrangements.

A Commercial Agreement grants additional negotiated rights. For an Organization above the
Revenue Threshold, it also covers Production Use. It is not required to discover whether
MIMIR works: evaluation is free for everyone (Section 5). See `COMMERCIAL-LICENSING.md`.

## 8. Distribution and MIMIR-Derived Models

You may Distribute the Model and MIMIR-Derived Models under this License. If You Distribute
model weights, You must include this License or a durable link to it, keep the attribution
notice in Section 4.5, state material modifications such as fine-tuning or quantization, and
not present the result as an official Mythologic product.

A MIMIR-Derived Model that contains or embodies MIMIR weights remains subject to this
License. An Adapter Artifact that does not contain Model weights may be licensed on other
terms.

## 9. Private Mythologic assets are not licensed

Receiving the Model does not grant access to, and this License does not license: private
training code; private training infrastructure; private datasets; unreleased checkpoints;
internal evaluation assets; proprietary research material; trade secrets; or future
unreleased models.

## 10. Trademarks

This License grants no right to use the MIMIR or Mythologic names, logos or other marks,
except for accurate, non-misleading statements that an application uses or is compatible with
MIMIR. See `TRADEMARKS.md`.

## 11. Third-party components

The Model's encoder is based on ModernBERT (`answerdotai/ModernBERT-Large-Instruct`), licensed
under the Apache License 2.0. That license and its notices remain applicable to the encoder
components and must be preserved in any distribution of the Model or a MIMIR-Derived Model to
the extent Apache-2.0 requires. Other third-party components remain subject to their own
terms.

## 12. No warranties; limitation of liability

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE MODEL IS PROVIDED “AS IS” AND “AS
AVAILABLE”, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, ACCURACY OR RELIABILITY. Licensor does
not warrant that outputs are accurate, complete, lawful or suitable for any particular
decision, and You are responsible for validating outputs and applying appropriate oversight.

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, LICENSOR WILL NOT BE LIABLE FOR INDIRECT,
INCIDENTAL, SPECIAL, CONSEQUENTIAL OR PUNITIVE DAMAGES, OR FOR LOST PROFITS, LOST REVENUE,
LOSS OF DATA OR BUSINESS INTERRUPTION, ARISING FROM THE MODEL OR THIS LICENSE. LICENSOR'S
TOTAL LIABILITY UNDER THIS LICENSE WILL NOT EXCEED THE GREATER OF US$100 OR THE FEES YOU PAID
TO LICENSOR FOR THE MODEL DURING THE TWELVE MONTHS BEFORE THE CLAIM. Nothing in this License
limits liability that applicable law does not allow to be limited.

You will indemnify Licensor against third-party claims arising from Your unlawful use of the
Model or Your breach of this License.

## 13. Term and termination

This License runs from acceptance until terminated. You accept it by downloading, using,
modifying or Distributing the Model. Licensor may terminate it for material breach if You do
not cure the breach within fifteen (15) calendar days of written notice, or immediately for
deliberate circumvention of Section 4 or for a violation that cannot be cured. On
termination, cease the conduct that breached the License; backups need not be deleted, and
outputs and decisions already produced need not be deleted. Provisions that by their nature
should survive termination do.

## 14. Licensing of releases

Each MIMIR release is governed by the license published with it. MIMIR-1 remains under the
MIMIR Model License unless Mythologic separately publishes a different license for that
model. This License does not convert to Apache-2.0, or to any other license, after a period
of time. Future releases may be published under different terms, and that does not change the
terms applicable to a release You already obtained.

## 15. Governing law; contact

This License is governed by the laws of France, excluding conflict-of-law rules, and without
prejudice to any mandatory rule of Your domicile. Subject to mandatory jurisdiction rules,
disputes are heard by the competent courts of Paris, France; a Licensee domiciled outside
France may bring proceedings in the courts of its own domicile. If a provision is held
invalid, it is enforced to the maximum extent permitted and the rest remains in effect.
Mandatory consumer and non-waivable rights are preserved.

Commercial licensing and legal notices:
[abderahmane.ainouche.ai@gmail.com](mailto:abderahmane.ainouche.ai@gmail.com), or through
[github.com/abderahmane-ai/mimir](https://github.com/abderahmane-ai/mimir).

---

**End of MIMIR Model License, Version 1.0.**

import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probability = 1
    for person in people:

        # Define if person has trait and how many genes it has
        trait = True if person in have_trait else False
        genes = n_genes(person, one_gene, two_genes)

        father = people[person]["father"]
        mother = people[person]["mother"] 
        prob_parents_passing = {father : 0, mother : 0}
        

        # If person has no parents, take the standard probabilities to have 0, 1 or 2 genes
        if father is None and mother is None:
            probability = probability * PROBS["gene"][genes]
        
        # If person has only father / only mother or none
        else:
            prob_parents_passing[father] = prob_get_gene_parent(father, one_gene, two_genes)
            prob_parents_passing[mother] = prob_get_gene_parent(mother, one_gene, two_genes)
            probability = probability * prob_given_parents(prob_parents_passing[father], prob_parents_passing[mother], genes)
        
        probability = probability * PROBS["trait"][genes][trait]
        
    return probability

def prob_get_gene_parent(parent, one_gene, two_genes):
    """
    Returns the probability that a parent passes the gene
    """
    if parent is None:
        return 0.5 * PROBS["gene"][1] + (1 - PROBS["mutation"]) * PROBS["gene"][2] + PROBS["mutation"] * PROBS["gene"][0]

    if parent in one_gene:
        return 0.5
    elif parent in two_genes:
        return (1 - PROBS["mutation"])
    else:
        return PROBS["mutation"]

def n_genes(person, one_gene, two_genes):
    """
    Return the number of genes a person has
    """
    genes = (
            1 if person in one_gene else
            2 if person in two_genes else
            0
        )
    return genes

def prob_given_parents(p_father, p_mother, genes):
    """
    Calculates probability of having a given quanity of genes if parents are known
    """
    p = (
            p_father * p_mother if genes == 2 else
            p_father * (1 - p_mother) + (1 - p_father) * p_mother if genes == 1 else
            (1 - p_father) * (1 - p_mother)
    )
    return p


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # Get the number of genes
        n = n_genes(person, one_gene, two_genes)

        # Get the trait
        trait = True if person in have_trait else False

        # Update probabilities
        probabilities[person]["gene"][n] += p
        probabilities[person]["trait"][trait] += p
        

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        
        # Normalise genes probability
        s = sum(list(probabilities[person]["gene"].values()))
        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] *=  1 / s

        # Normalise trait probability
        s = sum(list(probabilities[person]["trait"].values()))
        for trait in probabilities[person]["trait"]:
            probabilities[person]["trait"][trait] *=  1 / s

if __name__ == "__main__":
    main()
